# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Environment choices for official SGLang startup, resolved one setting at a time."""

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
SGLANG_HCU_LIGHTOP_PAGED_MQA_NATIVE_SCHEDULE
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


def automatic_values(args, family, env):
    """Inherit deployment inputs after explicit and file settings are combined."""
    auto = {}
    if family is None:
        return auto
    ep = args.moe_a2a_backend == "deepep" and args.ep_size != "1"
    pd = args.disaggregation_transfer_backend == "mooncake"
    decode = args.disaggregation_mode == "decode"
    fp8 = family == "fp8"
    w4 = family == "w4a8"
    hca = env.get("NCCL_IB_HCA")
    if "NCCL_SOCKET_IFNAME" in env:
        auto["GLOO_SOCKET_IFNAME"] = env["NCCL_SOCKET_IFNAME"]
    if ep and hca:
        auto["ROCSHMEM_ALLOWED_IBV_DEVICES"] = hca
    if pd and args.disaggregation_ib_device and (decode or w4):
        auto["MC_TE_FILTERS"] = args.disaggregation_ib_device
        if fp8:
            auto["MC_ALLOWED_IBV_DEVICES"] = args.disaggregation_ib_device
    if w4 and "NO_PROXY" in env:
        auto["no_proxy"] = env["NO_PROXY"]
    return auto


def resolve_environment(args, family, inherited, configured, keys):
    # Rules that depend on site inputs see the same explicit/config precedence.
    auto = automatic_values(args, family, {**configured, **inherited})
    # Preserve the channel-INT4 checkpoint contract even with a custom YAML.
    defaults = {"HCU_W4A8_MOE": "native"} if family == "w4a8" else {}
    env = inherited.copy()
    sources = {}
    for key in keys:
        for source, values in (("explicit", inherited), ("config", configured),
                               ("auto", auto), ("default", defaults)):
            if key in values:
                env[key] = str(values[key])
                sources[key] = source
                break
        # Otherwise leave the owning runtime's default in effect.
    return env, sources
