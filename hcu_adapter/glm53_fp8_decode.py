# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Launch the eight-node GLM-5.3 Channel-FP8 Decode recipe."""

import argparse
import json
import os
import sys
from pathlib import Path

# Set these in the child environment before glibc, HIP or SGLang starts.
DEFAULTS = {
    "SPT_NOENV": "1",
    "HCU_ADAPTER_HOST": "sglang",
    "SGLANG_PLUGINS": "hcu",
    "VLLM_PLUGINS": "none",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
    "SGLANG_ROCM_USE_MULTI_STREAM": "0",
    "SGLANG_SCHEDULER_SKIP_ALL_GATHER": "1",
    "SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH": "1",
    "SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT": "120",
    "NCCL_IB_DISABLE": "0",
    "MC_ENABLE_DEST_DEVICE_AFFINITY": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "ROCSHMEM_DISABLE_HDP_FLUSH": "1",
    "ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX": "288",
    "ROCSHMEM_IPC_MNVL": "1",
    "ROCSHMEM_IPC_MNVL_NPES": "32",
    "ROCSHMEM_BACKEND": "gda",
    "ROCSHMEM_GDA_PROVIDER": "shca",
    "DEEPEP_ENABLE_LL_LAYERED_OPT": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D": "1",
    "SGLANG_USE_DEEPGEMM_MOE": "1",
    "SGLANG_USE_FP8_W8A8_MOE": "1",
    "SGLANG_HCU_GLM_CT_BF16_ROUTER": "1",
    "SGLANG_USE_LIGHTOP": "1",
    "SGLANG_GLM_OPT_DECODE": "1",
    "SGLANG_GLM_HCU_GEMM": "1",
    "SGLANG_HCU_USE_LIGHTOP_ROPE": "1",
    "SGLANG_OPT_USE_TOPK_V2": "0",
    "SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q": "1",
    "SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK": "1",
    "SGLANG_DSA_FUSE_TOPK": "1",
    "LIGHTOP_SPARSE_MQA_GROUP6_ROWS_PER_CTA": "3",
    "ALLREDUCE_STREAM_WITH_COMPUTE": "1",
    "HIP_KERNEL_EVENT_SYSTENFENCE": "1",
    "GLIBC_TUNABLES": "glibc.rtld.optional_static_tls=0x40000",
    "HIP_KERNEL_BATCH_CEILING": "100",
    "GPU_FORCE_BLIT_COPY_SIZE": "16",
    "HSA_KERNARG_POOL_SIZE": "8388608",
    "ROC_AQL_QUEUE_SIZE": "131072",
    "HIP_BUFFER_EXTRA_SIZE": "0",
    "HIP_GRAPH_ACCUMULATE_DISPATCH": "1",
    "HIP_GRAPH_USE_CMD_CACHE": "1",
    "GPU_MAX_HW_QUEUES": "3",
}

SITE_KEYS = (
    "HIPBLASLT_TENSILE_LIBPATH",
    "HIPBLASLT_TUNING_OVERRIDE_FILE",
    "HIP_VISIBLE_DEVICES",
    "NCCL_IB_HCA",
    "NCCL_SOCKET_IFNAME",
    "ROCBLAS_TENSILE_LIBPATH",
)
DERIVED_KEYS = (
    "GLOO_SOCKET_IFNAME", "ROCSHMEM_ALLOWED_IBV_DEVICES",
    "MC_ALLOWED_IBV_DEVICES", "MC_TE_FILTERS",
)
CONFIG_KEYS = set(DEFAULTS).union(SITE_KEYS, DERIVED_KEYS)


def launch_options(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    expected = {
        "nnodes": "8", "tp-size": "32", "ep-size": "32", "attn-dp-size": "32",
        "dp-size": "1", "dcp-size": "1", "attn-cp-size": "1",
        "disaggregation-mode": "decode", "disaggregation-transfer-backend": "mooncake",
        "moe-a2a-backend": "deepep", "deepep-mode": "low_latency",
    }
    for name, value in expected.items():
        parser.add_argument("--" + name, choices=[value], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--disaggregation-ib-device", required=True)
    args, _ = parser.parse_known_args(argv)
    with (args.model_path / "config.json").open() as stream:
        model = json.load(stream)
    quant = model.get("quantization_config", {})
    groups = quant.get("config_groups", {})
    if not (
        model.get("model_type") == "glm_moe_dsa"
        and quant.get("quant_method") == "compressed-tensors"
        and groups
        and all(_channel_fp8(group) for group in groups.values())
    ):
        parser.error("expected a GLM MoE DSA compressed-tensors Channel-FP8 checkpoint")
    return args


def _channel_fp8(group):
    weights = group.get("weights") or {}
    acts = group.get("input_activations") or {}
    return (
        (weights.get("type"), weights.get("num_bits"), weights.get("strategy"))
        == ("float", 8, "channel")
        and (acts.get("type"), acts.get("num_bits"), acts.get("strategy"), acts.get("dynamic"))
        == ("float", 8, "token", True)
    )


def launch_environment(inherited, configured, pd_devices):
    env = inherited.copy()
    sources = {}

    def assign(key, fallback=None, source="default"):
        if key in inherited:
            sources[key] = "explicit"
        elif key in configured:
            env[key] = str(configured[key])
            sources[key] = "config"
        elif fallback is not None:
            env[key] = fallback
            sources[key] = source

    for key in SITE_KEYS:
        assign(key)
    for key, value in DEFAULTS.items():
        assign(key, value)
    assign("GLOO_SOCKET_IFNAME", env.get("NCCL_SOCKET_IFNAME"), "auto")
    assign("ROCSHMEM_ALLOWED_IBV_DEVICES", env.get("NCCL_IB_HCA"), "auto")
    assign("MC_ALLOWED_IBV_DEVICES", pd_devices, "auto")
    assign("MC_TE_FILTERS", pd_devices, "auto")
    return env, sources


def main():
    from . import hcu_config

    argv = sys.argv[1:]
    options = launch_options(argv)
    configured = hcu_config.load()["hosts"].get("sglang", {}).get("glm53_fp8_decode_dp32", {})
    env, sources = launch_environment(os.environ, configured, options.disaggregation_ib_device)
    print("GLM-5.3 FP8 Decode: " + " ".join(
        f"{k}={env[k]}({source})" for k, source in sorted(sources.items())
    ), file=sys.stderr, flush=True)
    os.execve(sys.executable, [sys.executable, "-m", "hcu_sglang", *argv], env)


if __name__ == "__main__":
    main()
