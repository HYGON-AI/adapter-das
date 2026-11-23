# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""LightOp expert projections for the GLM-5.3-Flash TP8 block-FP8 layout."""

from contextvars import ContextVar
from functools import wraps
import logging
from weakref import ref

import torch

from hcu_adapter import hcu_env, registry

_LOG = logging.getLogger(__name__)
_BLOCK_M = 16
_MODE = 500
_BLOCK_K = 128
_WEIGHT_PACK_K = 64
_FP8 = torch.float8_e4m3fn
_PACKED = {}


def _pack_experts(weight):
    experts, n, k = weight.shape
    # The LightOp blockwise expert ABI groups the raw E4M3 bytes by K64,
    # followed by the output column. The block scales keep their original order.
    return (
        weight.view(torch.uint8)
        .reshape(experts, n, k // _WEIGHT_PACK_K, _WEIGHT_PACK_K)
        .transpose(1, 2)
        .contiguous()
        .view(experts, k // _WEIGHT_PACK_K, n * _WEIGHT_PACK_K)
        .view(_FP8)
    )


def prepare_layer(layer):
    """Replace the storage of both loaded expert parameters with packed bytes."""
    up = layer.w13_weight
    down = layer.w2_weight
    experts = up.shape[0]
    if (
        up.device.type != "cuda" or experts not in (288, 289)
        or up.shape != (experts, 512, 4096) or down.shape != (experts, 4096, 256)
        or up.dtype != _FP8 or down.dtype != _FP8
        or not up.is_contiguous() or not down.is_contiguous()
        or getattr(up, "is_shuffled", False) or getattr(down, "is_shuffled", False)
        or not torch.cuda.get_device_properties(up.device).gcnArchName.startswith("gfx938")
    ):
        return False
    for weight, scale in (
        (up, layer.w13_weight_scale_inv),
        (down, layer.w2_weight_scale_inv),
    ):
        assert scale.device == weight.device
        assert scale.dtype == torch.float32
        assert scale.shape == (experts, weight.shape[1] // _BLOCK_K, weight.shape[2] // _BLOCK_K)

    for weight in (up, down):
        if _packed(weight) is not None:
            raise RuntimeError("LightOp expert weight storage was already packed")
        packed = _pack_experts(weight)
        shape = tuple(packed.shape)
        # The host runner reads the logical shape from the parameter. Its
        # storage now contains exclusively the bytes expected by LightOp.
        weight.data = packed.view(weight.shape)
        _PACKED[weight.data_ptr()] = (ref(weight), shape)
    return True


def _packed(weight):
    key = weight.data_ptr()
    entry = _PACKED.get(key)
    if entry is None:
        return None
    owner = entry[0]()
    if owner is None or owner.data_ptr() != key:
        _PACKED.pop(key, None)
        return None
    if (weight.shape != owner.shape or weight.stride() != owner.stride()
            or weight.dtype != owner.dtype or weight.device != owner.device):
        raise RuntimeError("Unsupported view of packed LightOp expert weight storage")
    return weight.view(entry[1])


def install():
    if torch.version.hip is None:
        return
    if hcu_env.resolve("HCU_FUSED_MOE_IMPL")[0] != "triton":
        raise RuntimeError("Packed LightOp experts require HCU_FUSED_MOE_IMPL=triton")

    from lightop import moe_gemm_marlin_w8a8_blockwise_fp8
    from sglang.srt.distributed.parallel_state import (
        get_moe_expert_parallel_world_size,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
    from sglang.srt.layers.quantization.fp8_utils import sglang_per_token_group_quant_fp8

    # The package also exports a function named fused_moe.
    from importlib import import_module

    fm = import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe")
    original_launch = fm.invoke_fused_moe_kernel
    if getattr(original_launch, "_hcu_lightop_block_fp8", False):
        return

    original_load = Fp8MoEMethod.process_weights_after_loading
    original_prepare = fm._prepare_fused_moe_run
    original_config = fm.try_get_optimal_moe_config
    active = ContextVar("hcu_lightop_expert_alignment", default=False)

    @wraps(original_load)
    def load(self, layer):
        original_load(self, layer)
        if not (
            self.block_quant and tuple(self.quant_config.weight_block_size or ()) == (128, 128)
            and get_tensor_model_parallel_world_size() == 8
            and get_moe_expert_parallel_world_size() == 1
        ):
            return
        runner = getattr(self, "runner", None)
        if (runner is None or not runner.runner_backend.is_triton() or runner.lora_enabled
                or getattr(runner.fused_func, "__name__", "") != "fused_experts_none_to_triton"
                or runner.config.apply_router_weight_on_input or runner.config.no_combine):
            return
        if prepare_layer(layer):
            _LOG.info("LightOp block FP8 expert weights packed in one storage: E=%d",
                      layer.w13_weight.shape[0])

    @wraps(original_config)
    def choose_configuration(*args, **kwargs):
        selected = original_config(*args, **kwargs)
        if not active.get():
            return selected
        config, (down, meta) = selected
        return (
            {**config, "BLOCK_SIZE_M": _BLOCK_M},
            ({**down, "BLOCK_SIZE_M": _BLOCK_M} if down else None, meta),
        )

    @wraps(original_prepare)
    def prepare(hidden_states, w1, w2, topk_ids, *args, **kwargs):
        packed_up, packed_down = _packed(w1), _packed(w2)
        if packed_up is None and packed_down is None:
            return original_prepare(hidden_states, w1, w2, topk_ids, *args, **kwargs)
        if (
            packed_up is None or packed_down is None or not kwargs.get("use_fp8_w8a8")
            or tuple(kwargs.get("block_shape") or ()) != (128, 128)
            or topk_ids.shape[-1] not in (8, 9)
        ):
            raise RuntimeError("Unsupported routing configuration for packed LightOp experts")
        token = active.set(True)
        try:
            return original_prepare(hidden_states, w1, w2, topk_ids, *args, **kwargs)
        finally:
            active.reset(token)

    @wraps(original_launch)
    def launch(
        A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids, sorted_token_ids,
        expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, *args, **kwargs,
    ):
        packed = _packed(B)
        if packed is None:
            return original_launch(
                A, B, bias, C, A_scale, B_scale, B_zp, topk_weights, topk_ids, sorted_token_ids,
                expert_ids, num_tokens_post_padded, mul_routed_weight, top_k, config, *args, **kwargs,
            )
        is_down = B.shape[-1] == 256
        unsupported = (
            "no_combine", "a_use_tma", "b_use_tma", "c_sorted", "fuse_sum_all_reduce",
            "fuse_add_to_output", "mask_output", "lora_preserve_base", "fuse_swiglu",
        )
        rows, routes = topk_ids.shape
        if rows == 0:
            return
        if (
            args or routes not in (8, 9)
            or config.get("BLOCK_SIZE_M") != _BLOCK_M
            or not kwargs.get("use_fp8_w8a8") or tuple(kwargs.get("block_shape") or ()) != (128, 128)
            or bias is not None or B_zp is not None
            or B_scale is None or B_scale.dtype != torch.float32
            or any(kwargs.get(option) for option in unsupported)
            or C.dtype != torch.bfloat16 or not C.is_contiguous()
            or C.numel() != rows * routes * B.shape[1]
            or topk_weights.dtype != torch.float32 or not topk_weights.is_contiguous()
            or sorted_token_ids.dtype != torch.int32 or expert_ids.dtype != torch.int32
            or num_tokens_post_padded.dtype != torch.int32
            or A.shape[0] < rows * (routes if is_down else 1)
            or (A.dtype == _FP8 and (A_scale is None or A_scale.dtype != torch.float32))
            or mul_routed_weight != is_down or top_k != (1 if is_down else routes)
        ):
            raise RuntimeError("Unsupported GEMM call for packed LightOp experts")
        input_rows = rows * (routes if is_down else 1)
        # LightOp reads the row count from A. A prequantized buffer may have
        # extra rows for alignment, while SGLang pads route IDs with rows*routes.
        A = A[:input_rows].contiguous()
        if A.dtype == _FP8:
            A_scale = A_scale[:input_rows]
        else:
            A, A_scale = sglang_per_token_group_quant_fp8(A, _BLOCK_K)
        moe_gemm_marlin_w8a8_blockwise_fp8(
            A, packed, C.view(rows, routes, B.shape[1]), A_scale, B_scale,
            topk_weights if is_down else None, sorted_token_ids, expert_ids, num_tokens_post_padded,
            top_k, {"MODE": _MODE, "DELTA": 1},
        )

    registry.register(
        "glm53_lightop_experts", launch, target=f"{fm.__name__}:invoke_fused_moe_kernel",
        mechanism=registry.MECH_SEAM,
        detail="LightOp gfx938 block-W8A8 expert projections, retaining SGLang routing and combination",
    )
    launch._hcu_lightop_block_fp8 = True
    Fp8MoEMethod.process_weights_after_loading = load
    fm._prepare_fused_moe_run = prepare
    fm.try_get_optimal_moe_config = choose_configuration
    fm.invoke_fused_moe_kernel = launch
