# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Environment defaults for the supported GLM-5.3 release launch recipes."""

from dataclasses import dataclass
from pathlib import Path


def preset_env(filename):
    from .hcu_config import _parse_file

    return _parse_file(str(Path(__file__).with_name("launch_configs") / filename))["hosts"]["sglang"]["env"]


@dataclass(frozen=True)
class Profile:
    filename: str
    site: tuple
    derived: tuple
    topology: dict
    offline: bool = False
    simulate_disabled: bool = False

    @property
    def defaults(self):
        # Compatibility launchers also exported these legacy policy values.
        values = preset_env(self.filename)
        if self.offline:
            values.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if self.simulate_disabled:
            values["SGLANG_SIMULATE_ACC_LEN"] = "-1"
        return values


W4_SITE = ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NO_PROXY")
FP8_SITE = ("HIP_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA")
FP8_BLAS_SITE = (*FP8_SITE, "HIPBLASLT_TENSILE_LIBPATH",
                 "HIPBLASLT_TUNING_OVERRIDE_FILE", "ROCBLAS_TENSILE_LIBPATH")
COLLECTIVES = ("GLOO_SOCKET_IFNAME", "ROCSHMEM_ALLOWED_IBV_DEVICES")

PROFILES = {
    "glm53_w4a8_ifb_tp8": Profile(
        filename="glm53-w4a8-ifb-tp8.yaml", site=W4_SITE, derived=('GLOO_SOCKET_IFNAME', 'no_proxy'),
        topology={
            "tp_size": '8',
            "quantization": 'compressed-tensors',
        },
    ),
    "glm53_w4a8_prefill_cp8": Profile(
        filename="glm53-w4a8-prefill.yaml", offline=True, site=W4_SITE, derived=(*COLLECTIVES, 'MC_TE_FILTERS', 'no_proxy'),
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "pp_size": '2',
            "ep_size": '8',
            "attn_cp_size": '8',
            "disaggregation_mode": 'prefill',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'normal',
            "enable_prefill_cp": True,
            "quantization": 'compressed-tensors',
            "disaggregation_transfer_backend": 'mooncake',
        },
    ),
    "glm53_w4a8_decode_dp16": Profile(
        filename="glm53-w4a8-decode.yaml", offline=True, site=W4_SITE, derived=(*COLLECTIVES, 'MC_TE_FILTERS', 'no_proxy'),
        topology={
            "nnodes": '2',
            "tp_size": '16',
            "ep_size": '16',
            "dp_size": '16',
            "disaggregation_mode": 'decode',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'low_latency',
            "enable_dp_attention": True,
            "quantization": 'compressed-tensors',
            "disaggregation_transfer_backend": 'mooncake',
        },
    ),
    "glm53_fp8_ifb_cp8": Profile(
        filename="glm53-fp8-ifb-cp8.yaml", offline=True, site=FP8_SITE, derived=COLLECTIVES,
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "ep_size": '8',
            "attn_cp_size": '8',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'normal',
            "enable_prefill_cp": True,
        },
    ),
    "glm53_fp8_ifb_dp8": Profile(
        filename="glm53-fp8-ifb-dp8.yaml", offline=True, simulate_disabled=True, site=FP8_BLAS_SITE, derived=COLLECTIVES,
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "ep_size": '8',
            "attn_dp_size": '8',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'auto',
        },
    ),
    "glm53_fp8_ifb_dcp4": Profile(
        filename="glm53-fp8-ifb-dcp4.yaml", offline=True, site=FP8_BLAS_SITE, derived=COLLECTIVES,
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "ep_size": '8',
            "attn_dp_size": '2',
            "dcp_size": '4',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'auto',
        },
    ),
    "glm53_fp8_ifb_hisparse": Profile(
        filename="glm53-fp8-ifb-dp8.yaml", offline=True, site=FP8_BLAS_SITE, derived=COLLECTIVES,
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "ep_size": '8',
            "attn_dp_size": '8',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'auto',
            "enable_hisparse": True,
        },
    ),
    "glm53_fp8_prefill_cp8": Profile(
        filename="glm53-fp8-prefill-cp8.yaml", offline=True, site=FP8_SITE, derived=COLLECTIVES,
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "ep_size": '8',
            "attn_cp_size": '8',
            "disaggregation_mode": 'prefill',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'normal',
            "enable_prefill_cp": True,
            "disaggregation_transfer_backend": 'mooncake',
        },
    ),
    "glm53_fp8_decode_dp8": Profile(
        filename="glm53-fp8-decode.yaml", site=FP8_BLAS_SITE, derived=(*COLLECTIVES, 'MC_ALLOWED_IBV_DEVICES', 'MC_TE_FILTERS'),
        topology={
            "nnodes": '2',
            "tp_size": '8',
            "ep_size": '8',
            "attn_dp_size": '8',
            "disaggregation_mode": 'decode',
            "moe_a2a_backend": 'deepep',
            "deepep_mode": 'low_latency',
            "disaggregation_transfer_backend": 'mooncake',
        },
    ),
}

# Supported deployment overrides, independent of the compatibility launchers.
LAUNCH_ENV_KEYS = frozenset("""
ALLREDUCE_STREAM_WITH_COMPUTE
DEEPEP_ENABLE_LL_LAYERED_OPT
DEEP_EP_NORMAL_MNVL
GLIBC_TUNABLES
GLOO_SOCKET_IFNAME
GPU_FORCE_BLIT_COPY_SIZE
GPU_MAX_HW_QUEUES
HCU_ADAPTER_HOST
HCU_DEEPEP_BRIDGE
HCU_FUSED_ADD_RMS_IMPL
HCU_RMS_NORM_IMPL
HCU_SLIMQUANT
HCU_W4A8_MOE
HF_HUB_OFFLINE
HIPBLASLT_TENSILE_LIBPATH
HIPBLASLT_TUNING_OVERRIDE_FILE
HIP_BUFFER_EXTRA_SIZE
HIP_GRAPH_ACCUMULATE_DISPATCH
HIP_GRAPH_USE_CMD_CACHE
HIP_KERNEL_BATCH_CEILING
HIP_KERNEL_EVENT_SYSTENFENCE
HIP_VISIBLE_DEVICES
HSA_KERNARG_POOL_SIZE
LIGHTOP_MQA_LOGITS_BETA
LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA
MC_ALLOWED_IBV_DEVICES
MC_ENABLE_DEST_DEVICE_AFFINITY
MC_GID_INDEX
MC_TE_FILTERS
NCCL_IB_DISABLE
NCCL_IB_GID_INDEX
NCCL_IB_HCA
NCCL_NET_PLUGIN
NCCL_PLUGIN_P2P
NCCL_SOCKET_IFNAME
NO_PROXY
OMP_NUM_THREADS
ROCBLAS_TENSILE_LIBPATH
ROCSHMEM_ALLOWED_IBV_DEVICES
ROCSHMEM_BACKEND
ROCSHMEM_DISABLE_HDP_FLUSH
ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX
ROCSHMEM_GDA_PROVIDER
ROCSHMEM_GDR_DISABLE_XDP
ROCSHMEM_IB_GID_INDEX
ROCSHMEM_IPC_MNVL
ROCSHMEM_IPC_MNVL_NPES
ROC_AQL_QUEUE_SIZE
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
SGLANG_DEQUANT_K_CACHE_BLOCK_T
SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT
SGLANG_DSA_FUSE_TOPK
SGLANG_DSA_HCU_INT8_INDEX_K_CACHE
SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK
SGLANG_DSA_HCU_REUSE_SORTED_TOPK
SGLANG_DSA_MQA_SPLIT_BY_SEQ
SGLANG_ENABLE_CP_V2
SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q
SGLANG_ENABLE_LOGPROB_CHUNK
SGLANG_GLM_HCU_GEMM
SGLANG_GLM_OPT_DECODE
SGLANG_GLM_OPT_PREFILL
SGLANG_GLM_PREFILL_SHARED_OVERLAP
SGLANG_HCU_GLM53_FP8_K1_AUTO_BUILD
SGLANG_HCU_GLM_CT_BF16_ROUTER
SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH
SGLANG_HCU_USE_LIGHTOP_ROPE
SGLANG_HCU_W8A8_USE_LIGHTOP
SGLANG_LIGHTOP_TOPK
SGLANG_LOGPROB_CHUNK_SIZE
SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D
SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH
SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB
SGLANG_OPT_USE_TOPK_V2
SGLANG_PLUGINS
SGLANG_ROCM_USE_AITER_MOE
SGLANG_ROCM_USE_MULTI_STREAM
SGLANG_SCHEDULER_SKIP_ALL_GATHER
SGLANG_SIMULATE_ACC_LEN
SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND
SGLANG_USE_AITER
SGLANG_USE_DEEPGEMM_MOE
SGLANG_USE_FP8_W8A8_MOE
SGLANG_USE_LIGHTOP
SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE
SGLANG_USE_W4A8_CONTIGUOUS_HIPC
SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT
SGLANG_W4A8_TPMOE_BACKEND
SPT_NOENV
TRANSFORMERS_OFFLINE
VLLM_PLUGINS
no_proxy
""".split())
