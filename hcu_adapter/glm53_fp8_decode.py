# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Launch the eight-node GLM-5.3 Channel-FP8 Decode recipe."""

import argparse
import json
import os
import sys
from pathlib import Path

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
def legacy_defaults():
    from .glm53_profiles import preset_env

    return {**preset_env("glm53-fp8-decode.yaml"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


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


def launch_environment(inherited, configured, pd_devices, *, defaults=None, site=SITE_KEYS, derived=DERIVED_KEYS):
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

    for key in site:
        assign(key)
    if defaults is None:
        defaults = legacy_defaults()
    for key, value in defaults.items():
        assign(key, value)
    automatic = {
        "GLOO_SOCKET_IFNAME": env.get("NCCL_SOCKET_IFNAME"),
        "ROCSHMEM_ALLOWED_IBV_DEVICES": env.get("NCCL_IB_HCA"),
        "MC_ALLOWED_IBV_DEVICES": pd_devices,
        "MC_TE_FILTERS": pd_devices,
        "no_proxy": env.get("NO_PROXY"),
    }
    for key in derived:
        assign(key, automatic[key], "auto")
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
