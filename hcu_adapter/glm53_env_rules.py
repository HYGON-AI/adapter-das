# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Environment choices for official SGLang startup, resolved one setting at a time."""

from .glm53_profiles import COMMON


def automatic_values(args, family, env):
    auto = {}
    if family is None:
        return auto

    fp8 = family == "fp8"
    w4 = family == "w4a8"
    decode = args.disaggregation_mode == "decode"
    prefill = args.disaggregation_mode == "prefill"
    ifb = args.disaggregation_mode in (None, "null")
    pd = args.disaggregation_transfer_backend == "mooncake" and (decode or prefill)
    ep = args.moe_a2a_backend == "deepep" and args.ep_size != "1"
    cp = args.enable_prefill_cp and args.attn_cp_size != "1"
    dp = args.attn_dp_size != "1" or args.enable_dp_attention
    local_tp = w4 and args.ep_size == "1" and args.tp_size == "8"
    fp8_dp = fp8 and dp
    w4_pd = w4 and pd and ep
    hca = env.get("NCCL_IB_HCA", "")
    shca = bool(hca) and all(n.startswith("shca_") for n in hca.split(","))

    if w4:
        auto.update(HCU_SLIMQUANT="0", HCU_W4A8_MOE="native", SGLANG_HCU_W8A8_USE_LIGHTOP="1",
                    SGLANG_ROCM_USE_AITER_MOE="0")
    if local_tp:
        auto.update(SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE="1", SGLANG_W4A8_TPMOE_BACKEND="lightop")
    if fp8:
        auto.update(SGLANG_USE_FP8_W8A8_MOE="1", SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D="1",
                    VLLM_PLUGINS="none")
    if ep:
        auto.update(SGLANG_USE_DEEPGEMM_MOE="1", SGLANG_HCU_USE_LIGHTOP_ROPE="1")
    if args.nnodes != "1":
        auto["NCCL_IB_DISABLE"] = "0"
    elif local_tp and ifb:
        auto["NCCL_IB_DISABLE"] = "1"
    if fp8 or pd:
        auto["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "1200"
    if w4 or fp8_dp:
        auto.update(HIP_GRAPH_ACCUMULATE_DISPATCH="1", SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q="1",
                    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT="120")
    if local_tp or cp:
        auto.update(SGLANG_DEQUANT_K_CACHE_BLOCK_T="16", SGLANG_DSA_MQA_SPLIT_BY_SEQ="1",
                    SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB="2")
    if local_tp or dp:
        auto.update(HIP_GRAPH_USE_CMD_CACHE="1", LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA="3",
                    SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK="1")
    if w4_pd:
        auto.update(HCU_DEEPEP_BRIDGE="0", HCU_FUSED_ADD_RMS_IMPL="lightop", HCU_RMS_NORM_IMPL="lightop",
                    OMP_NUM_THREADS="1", SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER="1",
                    SGLANG_DSA_HCU_INT8_INDEX_K_CACHE="1", SGLANG_LIGHTOP_TOPK="1", SGLANG_USE_AITER="0")
        if shca:
            auto.update(MC_GID_INDEX="0", NCCL_IB_GID_INDEX="0", NCCL_NET_PLUGIN="shca",
                        NCCL_PLUGIN_P2P="ib", ROCSHMEM_GDR_DISABLE_XDP="0", ROCSHMEM_IB_GID_INDEX="0")
    if w4 and prefill and cp:
        auto.update(HIP_GRAPH_USE_CMD_CACHE="0", SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE="0",
                    SGLANG_USE_W4A8_CONTIGUOUS_HIPC="1")
        if args.pp_size != "1":
            auto["SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH"] = "0"
    if w4_pd or fp8_dp:
        auto.update(SGLANG_HCU_GLM_CT_BF16_ROUTER="1", SGLANG_ROCM_USE_MULTI_STREAM="0")
    if pd and (w4 or decode):
        auto["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "1"
    if cp:
        auto.update(SGLANG_DSA_HCU_REUSE_SORTED_TOPK="1", SGLANG_ENABLE_CP_V2="1")
    if fp8 and cp:
        auto.update(LIGHTOP_MQA_LOGITS_BETA="1", SGLANG_GLM_OPT_PREFILL="1",
                    SGLANG_GLM_PREFILL_SHARED_OVERLAP="1", SGLANG_HCU_GLM53_FP8_K1_AUTO_BUILD="1")
        if ifb:
            auto.update(SGLANG_ENABLE_LOGPROB_CHUNK="1", SGLANG_LOGPROB_CHUNK_SIZE="2048")
    if fp8 and ep and args.deepep_mode in ("normal", "auto"):
        auto["DEEP_EP_NORMAL_MNVL"] = "1"
    if w4_pd or fp8:
        auto["SGLANG_DSA_FUSE_TOPK"] = "1" if args.dcp_size == "1" else "0"
    if fp8 and (dp or prefill):
        auto["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "python"
    if dp:
        auto["GPU_MAX_HW_QUEUES"] = "3"
        if ep and shca:
            auto.update(ROCSHMEM_BACKEND="gda", ROCSHMEM_GDA_PROVIDER="shca")
    if dp and decode and ep and args.deepep_mode == "low_latency":
        auto["SGLANG_SCHEDULER_SKIP_ALL_GATHER"] = "1"
        auto["DEEPEP_ENABLE_LL_LAYERED_OPT"] = "0" if fp8 else "1"
    if fp8_dp:
        auto.update(HIP_BUFFER_EXTRA_SIZE="0", SGLANG_GLM_HCU_GEMM="1", SGLANG_GLM_OPT_DECODE="1",
                    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="1", SPT_NOENV="1")
        if ep:
            auto.update(DEEPEP_ENABLE_LL_LAYERED_OPT="0", SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="256")
            # The SHMEM tuning is evidenced on four-device nodes at TP8 or TP32.
            if shca and (args.nnodes, args.tp_size) in {("2", "8"), ("8", "32")}:
                auto.update(ROCSHMEM_IPC_MNVL="1", ROCSHMEM_IPC_MNVL_NPES="32")
        if ifb:
            auto["SGLANG_SCHEDULER_SKIP_ALL_GATHER"] = "0"
    if fp8 and ep and shca:
        auto.update(ROCSHMEM_DISABLE_HDP_FLUSH="1", ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX="288")
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
    defaults = COMMON if family else {}
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
