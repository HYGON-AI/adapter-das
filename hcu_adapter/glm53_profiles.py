# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Environment defaults for the supported GLM-5.3 release launch recipes."""

from dataclasses import dataclass

COMMON = {
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "GPU_FORCE_BLIT_COPY_SIZE": "16",
    "HCU_ADAPTER_HOST": "sglang",
    "HIP_KERNEL_BATCH_CEILING": "100",
    "HIP_KERNEL_EVENT_SYSTENFENCE": "1",
    "HSA_KERNARG_POOL_SIZE": "8388608",
    "ROC_AQL_QUEUE_SIZE": "131072",
    "SGLANG_OPT_USE_TOPK_V2": "0",
    "SGLANG_PLUGINS": "hcu",
    "SGLANG_USE_LIGHTOP": "1",
}

W4A8 = {
    **COMMON,
    "HCU_SLIMQUANT": "0",
    "HCU_W4A8_MOE": "native",
    "HIP_GRAPH_ACCUMULATE_DISPATCH": "1",
    "SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q": "1",
    "SGLANG_HCU_W8A8_USE_LIGHTOP": "1",
    "SGLANG_ROCM_USE_AITER_MOE": "0",
    "SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT": "120",
}

W4A8_PD = {
    **W4A8,
    "HCU_DEEPEP_BRIDGE": "0",
    "HCU_FUSED_ADD_RMS_IMPL": "lightop",
    "HCU_RMS_NORM_IMPL": "lightop",
    "HF_HUB_OFFLINE": "1",
    "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
    "MC_GID_INDEX": "0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_IB_GID_INDEX": "0",
    "NCCL_NET_PLUGIN": "shca",
    "NCCL_PLUGIN_P2P": "ib",
    "OMP_NUM_THREADS": "1",
    "ROCSHMEM_GDR_DISABLE_XDP": "0",
    "ROCSHMEM_IB_GID_INDEX": "0",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_HCU_INT8_INDEX_K_CACHE": "1",
    "SGLANG_HCU_GLM_CT_BF16_ROUTER": "1",
    "SGLANG_HCU_USE_LIGHTOP_ROPE": "1",
    "SGLANG_LIGHTOP_TOPK": "1",
    "SGLANG_ROCM_USE_MULTI_STREAM": "0",
    "SGLANG_USE_AITER": "0",
    "SGLANG_USE_DEEPGEMM_MOE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

W4A8_PREFILL_CP8 = {
    **W4A8_PD,
    "HIP_GRAPH_USE_CMD_CACHE": "0",
    "SGLANG_DEQUANT_K_CACHE_BLOCK_T": "16",
    "SGLANG_DSA_HCU_REUSE_SORTED_TOPK": "1",
    "SGLANG_DSA_MQA_SPLIT_BY_SEQ": "1",
    "SGLANG_ENABLE_CP_V2": "1",
    "SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH": "0",
    "SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB": "2",
    "SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE": "0",
    "SGLANG_USE_W4A8_CONTIGUOUS_HIPC": "1",
}

W4A8_DECODE_DP16 = {
    **W4A8_PD,
    "DEEPEP_ENABLE_LL_LAYERED_OPT": "1",
    "GPU_MAX_HW_QUEUES": "3",
    "HIP_GRAPH_USE_CMD_CACHE": "1",
    "LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA": "3",
    "ROCSHMEM_BACKEND": "gda",
    "ROCSHMEM_GDA_PROVIDER": "shca",
    "SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK": "1",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
}

W4A8_IFB_TP8 = {
    **W4A8,
    "HIP_GRAPH_USE_CMD_CACHE": "1",
    "LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA": "3",
    "NCCL_IB_DISABLE": "1",
    "SGLANG_DEQUANT_K_CACHE_BLOCK_T": "16",
    "SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK": "1",
    "SGLANG_DSA_MQA_SPLIT_BY_SEQ": "1",
    "SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB": "2",
    "SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE": "1",
    "SGLANG_W4A8_TPMOE_BACKEND": "lightop",
}

FP8 = {
    **COMMON,
    "NCCL_IB_DISABLE": "0",
    "ROCSHMEM_DISABLE_HDP_FLUSH": "1",
    "ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX": "288",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_HCU_USE_LIGHTOP_ROPE": "1",
    "SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D": "1",
    "SGLANG_USE_DEEPGEMM_MOE": "1",
    "SGLANG_USE_FP8_W8A8_MOE": "1",
    "VLLM_PLUGINS": "none",
}

FP8_CP8 = {
    **FP8,
    "DEEP_EP_NORMAL_MNVL": "1",
    "HF_HUB_OFFLINE": "1",
    "LIGHTOP_MQA_LOGITS_BETA": "1",
    "SGLANG_DEQUANT_K_CACHE_BLOCK_T": "16",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_DSA_HCU_REUSE_SORTED_TOPK": "1",
    "SGLANG_DSA_MQA_SPLIT_BY_SEQ": "1",
    "SGLANG_ENABLE_CP_V2": "1",
    "SGLANG_GLM_OPT_PREFILL": "1",
    "SGLANG_GLM_PREFILL_SHARED_OVERLAP": "1",
    "SGLANG_HCU_GLM53_FP8_K1_AUTO_BUILD": "1",
    "SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB": "2",
    "TRANSFORMERS_OFFLINE": "1",
}

FP8_IFB_CP8 = {
    **FP8_CP8,
    "SGLANG_ENABLE_LOGPROB_CHUNK": "1",
    "SGLANG_LOGPROB_CHUNK_SIZE": "2048",
}

FP8_PREFILL_CP8 = {
    **FP8_CP8,
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
}

FP8_ATTENTION_DP = {
    **FP8,
    "DEEPEP_ENABLE_LL_LAYERED_OPT": "0",
    "GPU_MAX_HW_QUEUES": "3",
    "HIP_BUFFER_EXTRA_SIZE": "0",
    "HIP_GRAPH_ACCUMULATE_DISPATCH": "1",
    "HIP_GRAPH_USE_CMD_CACHE": "1",
    "LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA": "3",
    "ROCSHMEM_BACKEND": "gda",
    "ROCSHMEM_GDA_PROVIDER": "shca",
    "ROCSHMEM_IPC_MNVL": "1",
    "ROCSHMEM_IPC_MNVL_NPES": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK": "1",
    "SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q": "1",
    "SGLANG_GLM_HCU_GEMM": "1",
    "SGLANG_GLM_OPT_DECODE": "1",
    "SGLANG_HCU_GLM_CT_BF16_ROUTER": "1",
    "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH": "1",
    "SGLANG_ROCM_USE_MULTI_STREAM": "0",
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    "SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT": "120",
    "SPT_NOENV": "1",
}

FP8_IFB_DP8 = {
    **FP8_ATTENTION_DP,
    "DEEP_EP_NORMAL_MNVL": "1",
    "HF_HUB_OFFLINE": "1",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "0",
    "SGLANG_SIMULATE_ACC_LEN": "-1",
    "TRANSFORMERS_OFFLINE": "1",
}

FP8_IFB_DCP4 = {
    **FP8_ATTENTION_DP,
    "DEEP_EP_NORMAL_MNVL": "1",
    "HF_HUB_OFFLINE": "1",
    "SGLANG_DSA_FUSE_TOPK": "0",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "0",
    "TRANSFORMERS_OFFLINE": "1",
}

FP8_IFB_HISPARSE = {
    **FP8_ATTENTION_DP,
    "DEEP_EP_NORMAL_MNVL": "1",
    "HF_HUB_OFFLINE": "1",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "0",
    "TRANSFORMERS_OFFLINE": "1",
}

FP8_DECODE_DP8 = {
    **FP8_ATTENTION_DP,
    "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
}



@dataclass(frozen=True)
class Profile:
    defaults: dict
    site: tuple
    derived: tuple
    topology: dict


W4_SITE = ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NO_PROXY")
FP8_SITE = ("HIP_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA")
FP8_BLAS_SITE = (*FP8_SITE, "HIPBLASLT_TENSILE_LIBPATH",
                 "HIPBLASLT_TUNING_OVERRIDE_FILE", "ROCBLAS_TENSILE_LIBPATH")
COLLECTIVES = ("GLOO_SOCKET_IFNAME", "ROCSHMEM_ALLOWED_IBV_DEVICES")

PROFILES = {
    "glm53_w4a8_ifb_tp8": Profile(
        defaults=W4A8_IFB_TP8, site=W4_SITE, derived=('GLOO_SOCKET_IFNAME', 'no_proxy'),
        topology={
            "tp_size": '8',
            "quantization": 'compressed-tensors',
        },
    ),
    "glm53_w4a8_prefill_cp8": Profile(
        defaults=W4A8_PREFILL_CP8, site=W4_SITE, derived=(*COLLECTIVES, 'MC_TE_FILTERS', 'no_proxy'),
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
        defaults=W4A8_DECODE_DP16, site=W4_SITE, derived=(*COLLECTIVES, 'MC_TE_FILTERS', 'no_proxy'),
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
        defaults=FP8_IFB_CP8, site=FP8_SITE, derived=COLLECTIVES,
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
        defaults=FP8_IFB_DP8, site=FP8_BLAS_SITE, derived=COLLECTIVES,
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
        defaults=FP8_IFB_DCP4, site=FP8_BLAS_SITE, derived=COLLECTIVES,
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
        defaults=FP8_IFB_HISPARSE, site=FP8_BLAS_SITE, derived=COLLECTIVES,
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
        defaults=FP8_PREFILL_CP8, site=FP8_SITE, derived=COLLECTIVES,
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
        defaults=FP8_DECODE_DP8, site=FP8_BLAS_SITE, derived=(*COLLECTIVES, 'MC_ALLOWED_IBV_DEVICES', 'MC_TE_FILTERS'),
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
LAUNCH_ENV_KEYS = frozenset().union(
    *(set(p.defaults).union(p.site, p.derived) for p in PROFILES.values())
)
