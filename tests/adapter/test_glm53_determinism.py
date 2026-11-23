# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 determinism contracts with CPU tensors and host/kernel substitutes."""

import importlib
import importlib.util
from itertools import product
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import _dynamo

from hcu_adapter import registry
from hcu_adapter.bridges import sglang_allreduce


_ROOT = Path(__file__).resolve().parents[2]
_FEATURES = "hcu_adapter.features.glm53_determinism"


class _HipTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class _LaneTensor(torch.Tensor):
    def to(self, dtype, *, fp_downcast_rounding=None):
        if dtype == torch.bfloat16 and self.dtype == torch.float32:
            assert fp_downcast_rounding == "rtne"
        return super().to(dtype)


class _Pointer:
    def __init__(self, data, offsets=0):
        self.data = data
        self.offsets = offsets

    def __add__(self, offset):
        return _Pointer(self.data, self.offsets + offset)

    @classmethod
    def from_tensor(cls, tensor):
        size = tensor.untyped_storage().nbytes() // tensor.element_size()
        flat = tensor.as_strided((size,), (1,), storage_offset=0)
        return cls(flat, tensor.storage_offset())


class _Launch:
    def __init__(self, run):
        self.run = run
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **meta):
            self.calls.append((grid, meta))
            self.run(grid, *args, **meta)

        return launch


@pytest.fixture
def cpu_host(monkeypatch):
    """Load the actual adapter code without importing an accelerator runtime."""
    staged = {}

    def module(name, **attrs):
        if name not in staged:
            value = ModuleType(name)
            value.__path__ = []
            staged[name] = value
            monkeypatch.setitem(sys.modules, name, value)
            parent, _, name_part = name.rpartition(".")
            if parent:
                setattr(module(parent), name_part, value)
        value = staged[name]
        for key, attr in attrs.items():
            setattr(value, key, attr)
        return value

    def load(name):
        path = _ROOT.joinpath(*name.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        return value

    def read(ptr, mask, other):
        positions = torch.where(mask, ptr.offsets, 0)
        return torch.where(mask, ptr.data[positions], other).as_subclass(_LaneTensor)

    def write(ptr, value, mask):
        ptr.data[ptr.offsets[mask]] = value[mask].as_subclass(torch.Tensor)

    def dot(a, b, acc, *, input_precision):
        assert input_precision == "ieee"
        return acc + a @ b

    pid = []
    triton = module("triton", cdiv=lambda n, tile: (n + tile - 1) // tile,
                    jit=lambda fn=None, **kwargs: fn if fn is not None else lambda f: f)
    tl = module("triton.language", constexpr=int, program_id=lambda axis: torch.tensor(pid[axis]),
                int64=torch.int64, float32=torch.float32, bfloat16=torch.bfloat16,
                cdiv=triton.cdiv, static_range=range, arange=torch.arange, load=read,
                store=write, zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
                dot=dot)

    def emulate(fn, pointer_count):
        def run(grid, *args, **meta):
            operands = tuple(_Pointer.from_tensor(tensor) for tensor in args[:pointer_count])
            dimensions = tuple(torch.tensor(value) for value in args[pointer_count:])
            tile = {key: value for key, value in meta.items() if key.isupper()}
            for indices in product(*(range(blocks) for blocks in grid)):
                pid[:] = indices
                fn(*operands, *dimensions, **tile)

        return _Launch(run)

    monkeypatch.setattr(torch.version, "hip", "cpu-contract")
    monkeypatch.setattr(registry, "register", lambda *args, **kwargs: None)
    return SimpleNamespace(module=module, load=load, emulate=emulate, tl=tl)


@pytest.mark.parametrize("contiguous", [True, False])
def test_tp8_reduce_executes_rank_order_and_keeps_fallback(cpu_host, monkeypatch, contiguous):
    kernel_name = "hcu_adapter.kernels.ordered_tp_sum_triton"
    kernel = cpu_host.load(kernel_name)
    launch = cpu_host.emulate(kernel._sum_bf16_tp8, 2)
    monkeypatch.setattr(kernel, "_sum_bf16_tp8", launch)
    monkeypatch.setitem(sys.modules, kernel_name, kernel)
    kernels = importlib.import_module("hcu_adapter.kernels")
    monkeypatch.setattr(kernels, "ordered_tp_sum_triton", kernel, raising=False)

    class Group:
        world_size = 8

        def __init__(self):
            self.fallback_inputs = []

        def _all_reduce_in_place(self, value) -> None:
            self.fallback_inputs.append(value)

        def _all_gather_into_tensor(self, output, _input):
            output = output.view_as(shards)
            for rank in (2, 7, 0, 3, 6, 5, 1, 4):
                output[rank].copy_(shards[rank])

    parallel = cpu_host.module("sglang.srt.distributed.parallel_state", GroupCoordinator=Group)
    tp, other = Group(), Group()
    parallel.get_tp_group = parallel.get_attn_tp_group = lambda: tp
    sglang_allreduce.install_ordered_tp_reduce()
    hook = Group._all_reduce_in_place
    sglang_allreduce.install_ordered_tp_reduce()
    assert Group._all_reduce_in_place is hook

    out = torch.zeros((3, 91), dtype=torch.bfloat16).as_subclass(_HipTensor)
    if not contiguous:
        out = out.T
    shards = torch.zeros((8, out.numel()), dtype=out.dtype)
    shards[0].fill_(256)
    shards[1].fill_(1)
    shards[2].fill_(-256)
    shards[7, -1] = 2
    expected = shards[0]
    for shard in shards[1:]:
        expected = (expected.float() + shard.float()).bfloat16()
    assert not torch.equal(expected, shards.float().sum(0).bfloat16())
    assert tp._all_reduce_in_place(out) is None
    torch.testing.assert_close(out.flatten(), expected, rtol=0, atol=0)
    assert launch.calls == [((2,), {"BLOCK": 256, "num_warps": 4})]

    cpu = torch.zeros(1, dtype=torch.bfloat16)
    fp32 = out.float()
    assert tp._all_reduce_in_place(cpu) is None
    assert tp._all_reduce_in_place(fp32) is None
    assert other._all_reduce_in_place(out) is None
    assert tp._all_reduce_in_place(out[:0]) is None
    assert len(tp.fallback_inputs) == 2
    assert tp.fallback_inputs[0] is cpu
    assert tp.fallback_inputs[1] is fp32
    assert len(other.fallback_inputs) == 1
    assert other.fallback_inputs[0] is out
    assert len(launch.calls) == 1


def _upstream_matmul(_grid, lhs, rhs, output, *args, **meta):
    output.copy_((lhs.float() @ rhs.float()).to(output.dtype))


def test_kda_projections_use_fixed_tiles_for_supported_shapes(cpu_host):
    class UnquantizedLinearMethod:
        def apply(self, layer, x, bias=None):
            return x @ layer.weight.T if bias is None else x @ layer.weight.T + bias

    def linear():
        return SimpleNamespace(
            weight=torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16),
            quant_method=UnquantizedLinearMethod(),
        )

    class Attention:
        def __init__(self, fused=True):
            self.do_fuse_qkvbfg = fused
            self.fused_qkvbfg_a_proj = linear()
            self.o_proj = linear()
            gate = torch.tensor([[[1, 2], [-1, 3]]], dtype=torch.bfloat16)
            self.fused_fg_b_proj = SimpleNamespace(
                weight=gate, forward=lambda x: x @ gate.transpose(-1, -2),
            )

    mm, bmm = _Launch(_upstream_matmul), _Launch(_upstream_matmul)
    cpu_host.module("sglang.srt.batch_invariant_ops.batch_invariant_ops",
                    bmm_kernel_persistent=bmm, matmul_kernel_persistent=mm)
    cpu_host.module("sglang.srt.layers.quantization.unquant", UnquantizedLinearMethod=UnquantizedLinearMethod)
    cpu_host.module("sglang.srt.models.glm5_next", Glm5NextLinearAttention=Attention)
    cpu_host.module("sglang.srt.utils.common", get_device_core_count=lambda: 8)
    cpu_host.load(f"{_FEATURES}.kda_invariant_projection").install()
    x = torch.tensor([[2, 1]], dtype=torch.bfloat16).as_subclass(_HipTensor)
    attn = Attention()
    for linear_layer in (attn.fused_qkvbfg_a_proj, attn.o_proj):
        output = linear_layer.quant_method.apply(linear_layer, x)
        torch.testing.assert_close(output, torch.tensor([[4, 10]], dtype=x.dtype))
    gate = attn.fused_fg_b_proj.forward(x.unsqueeze(0))
    torch.testing.assert_close(gate, torch.tensor([[[4, 1]]], dtype=x.dtype))
    tile = dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8)
    for grid, meta in mm.calls + bmm.calls:
        assert grid == (1,)
        assert {k: meta[k] for k in tile} == tile
    assert (len(mm.calls), len(bmm.calls)) == (2, 1)

    layer = attn.o_proj
    torch.testing.assert_close(layer.quant_method.apply(layer, x, x), torch.tensor([[6, 11]], dtype=x.dtype))
    assert layer.quant_method.apply(layer, x[:0]).shape == (0, 2)
    unfused = Attention(fused=False).o_proj
    torch.testing.assert_close(unfused.quant_method.apply(unfused, x), torch.tensor([[4, 10]], dtype=x.dtype))
    assert (len(mm.calls), len(bmm.calls)) == (2, 1)


def test_zero_rope_mla_fp32_tile_and_bf16_projection_dispatch(cpu_host, monkeypatch):
    def attend(qp, kv, *args):
        return torch.bmm(qp, kv)

    class Attention:
        use_dsa = True
        qk_rope_head_dim = 0

        def __init__(self, weight):
            self.w_kc = self.w_vc = weight

        def forward_absorb_prepare(self, x, out):
            return torch.bmm(x, self.w_kc, out=out)

        def forward_absorb_core(self, x):
            return torch.bmm(x, self.w_vc)

    bmm = _Launch(_upstream_matmul)
    cpu_host.module("sglang.srt.batch_invariant_ops.batch_invariant_ops", bmm_kernel_persistent=bmm)
    cpu_host.module("sglang.srt.models.deepseek_v2", DeepseekV2AttentionMLA=Attention)
    cpu_host.module("sglang.srt.utils.common", get_device_core_count=lambda: 8)
    flash = cpu_host.module("sgl_kernel.flash_attn", _attend_chunked=attend)
    mla = cpu_host.load(f"{_FEATURES}.glm53_invariant_mla")
    fixed = cpu_host.emulate(mla._bmm_fp32, 3)
    monkeypatch.setattr(mla, "_bmm_fp32", fixed)
    mla.install()

    q = torch.tensor([[[2.0], [-1.0]]]).as_subclass(_HipTensor)
    kv = torch.tensor([[[3.0, -4.0]]])
    expected = torch.tensor([[[6.0, -8.0], [-3.0, 4.0]]])
    args = (q, kv, None, None, 1.0, True, None, 0, None, None, torch.bfloat16, 2)
    torch.testing.assert_close(flash._attend_chunked(*args, qvp=None), expected)
    assert fixed.calls == []
    qv = SimpleNamespace(shape=(1, 2, 512))
    torch.testing.assert_close(flash._attend_chunked(*args, qvp=qv), expected)
    assert fixed.calls == [((1, 1), dict(BM=16, BN=32, BK=16, num_warps=4, num_stages=1))]

    attn = Attention(kv.bfloat16())
    out = torch.empty_like(expected, dtype=torch.bfloat16)
    assert attn.forward_absorb_prepare(q.bfloat16(), out) is out
    torch.testing.assert_close(out, expected.bfloat16())
    torch.testing.assert_close(attn.forward_absorb_core(q.bfloat16()), expected.bfloat16())
    assert [meta["BLOCK_SIZE_K"] for _, meta in bmm.calls] == [32, 32]
    attn.qk_rope_head_dim = 1
    torch.testing.assert_close(attn.forward_absorb_core(q.bfloat16()), expected.bfloat16())
    assert len(bmm.calls) == 2

    # Exercise a second reduction tile, distinct batch strides and a transposed RHS.
    lhs = (torch.arange(2 * 3 * 19, dtype=torch.float32) % 7 - 3).view(2, 3, 19)
    rhs = (torch.arange(2 * 5 * 19, dtype=torch.float32) % 5 - 2).view(2, 5, 19).transpose(-1, -2)
    torch.testing.assert_close(mla.fp32_bmm(lhs, rhs), torch.bmm(lhs, rhs), rtol=0, atol=0)
    assert fixed.calls[-1][0] == (1, 2)


def test_mhc_fixed_fp32_statistics_keep_batch_row_and_fallback(cpu_host, monkeypatch):
    def stats(x, weight, order):
        terms = x.float()[:, None, :] * weight[None, :, :]
        dots = ((terms[..., order[0]] + terms[..., order[1]]) + terms[..., order[2]]) + terms[..., 3]
        return dots / x.float().square().mean(-1, keepdim=True).sqrt()

    original_calls = []
    fixed_calls = []

    def original(x, weight, *_args):
        original_calls.append(x.shape)
        return None, None, 2 * torch.sigmoid(stats(x, weight, (0, 2, 1))[:, 2:4]), False

    def fixed(x, weight, _eps):
        fixed_calls.append(x.shape)
        return stats(x, weight, (0, 1, 2))

    mhc = cpu_host.module("sglang.kernels.ops.layernorm.mhc", _use_aiter_mhc=lambda: False,
                          _use_tilelang_mhc_pre=lambda: False, _is_hip=True,
                          _HC_MIX_DOT_PRECISION="ieee", _HC_MIX_BLOCK_K=4, hc_mix_stats=fixed)
    model = cpu_host.module("sglang.srt.models.glm5_next", _hc_pre_fn=original)
    monkeypatch.setattr(_dynamo, "disable", lambda fn: fn)
    cpu_host.load(f"{_FEATURES}.mhc_invariant_stats").install()

    x = torch.tensor([[1, 1, 1, 1], [1, 1, -1, 1]], dtype=torch.bfloat16).as_subclass(_HipTensor)
    weight = torch.zeros((8, 4))
    weight[2] = torch.tensor([2**24, 1, -(2**24), 0], dtype=torch.float32)
    args = (weight, torch.ones(3), torch.zeros(8), 2, 0.0, 0.0, 2)
    hidden, comb, post, flag = model._hc_pre_fn(x[:1], *args)
    torch.testing.assert_close(hidden, torch.ones((1, 2), dtype=x.dtype))
    torch.testing.assert_close(comb, torch.full((1, 4), 0.5))
    torch.testing.assert_close(post, torch.ones((1, 2)))
    assert flag is False
    for single, packed in zip((hidden, comb, post), model._hc_pre_fn(x, *args)[:3]):
        torch.testing.assert_close(single, packed[:1], rtol=0, atol=0)
    assert original_calls == []
    assert len(fixed_calls) == 2

    mhc._use_aiter_mhc = lambda: True
    _, _, fallback, _ = model._hc_pre_fn(x[:1], *args)
    torch.testing.assert_close(fallback, torch.tensor([[2 * torch.sigmoid(torch.tensor(1.0)).item(), 1]]))
    assert not torch.equal(fallback, post)
    assert len(original_calls) == 1
    assert len(fixed_calls) == 2
