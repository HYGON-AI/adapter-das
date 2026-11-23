# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Apply configured GLM-5.3 launch settings before SGLang starts its workers."""

import argparse
import json
import os
import sys
from pathlib import Path

from . import hcu_config, hcu_env
from .glm53_env_rules import LAUNCH_ENV_KEYS, resolve_environment


def _channel_fp8(group):
    weights = group.get("weights") or {}
    acts = group.get("input_activations") or {}
    return (
        (weights.get("type"), weights.get("num_bits"), weights.get("strategy"))
        == ("float", 8, "channel")
        and (acts.get("type"), acts.get("num_bits"), acts.get("strategy"), acts.get("dynamic"))
        == ("float", 8, "token", True)
    )


def checkpoint_family(path):
    with (path / "config.json").open() as stream:
        model = json.load(stream)
    if (model.get("hidden_size"), model.get("num_hidden_layers"), model.get("n_routed_experts")) != (6144, 78, 256):
        return None
    quant = model.get("quantization_config") or {}
    groups = (quant.get("config_groups") or {}).values()
    if model.get("model_type") != "glm_moe_dsa" or quant.get("quant_method") != "compressed-tensors" or not groups:
        return None
    if all(_channel_fp8(group) for group in groups):
        return "fp8"
    found_experts = False
    for group in groups:
        w = group.get("weights") or {}
        a = group.get("input_activations") or {}
        if (w.get("type"), w.get("strategy"), w.get("num_bits")) not in (
            ("int", "channel", 4), ("int", "channel", 8)
        ) or (a.get("type"), a.get("num_bits"), a.get("strategy"), a.get("dynamic")) != ("int", 8, "token", True):
            return None
        if w["num_bits"] == 4 and any("experts" in target for target in group.get("targets", [])):
            found_experts = True
    return "w4a8" if found_experts else None


def parse_options(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--disaggregation-ib-device")
    parser.add_argument("--ep-size", default="1")
    parser.add_argument("--disaggregation-mode")
    parser.add_argument("--disaggregation-transfer-backend")
    parser.add_argument("--moe-a2a-backend")
    args, _ = parser.parse_known_args(argv)
    return args


def apply_official_defaults(argv, original):
    # exec keeps the PID and makes loader and device defaults visible from
    # process creation. Consume the marker before SGLang spawns any workers.
    marker = "_HCU_GLM53_REEXEC_PID"
    if os.environ.pop(marker, None) == str(os.getpid()):
        return
    if "--help" in argv or "-h" in argv:
        return
    args = parse_options(argv)
    config = hcu_config.load()
    settings = config["hosts"].get("sglang", {})
    configured = {**config["overrides"], **settings.get("env", {})}
    family = None
    plugins = os.environ.get("SGLANG_PLUGINS", configured.get("SGLANG_PLUGINS"))
    hcu_allowed = plugins is None or "hcu" in {p.strip() for p in str(plugins).split(",")}
    if hcu_allowed and args.model_path and (args.model_path / "config.json").is_file():
        family = checkpoint_family(args.model_path)
    preset, _ = hcu_env.resolve("HCU_CONFIG")
    if preset.startswith("builtin:glm53-"):
        expected_family = preset.split("-", 2)[1]
        if family != expected_family:
            raise SystemExit(f"{preset} requires a local GLM-5.3 {expected_family} checkpoint and HCU plugin")
    env, sources = resolve_environment(args, family, os.environ, configured, LAUNCH_ENV_KEYS)
    if family == "w4a8" and env["HCU_W4A8_MOE"] != "native":
        raise SystemExit("GLM-5.3 W4A8 requires native MoE; check HCU_W4A8_MOE")
    if sources:
        print("HCU SGLang env: " + " ".join(
            f"{key}={env[key]}({source})" for key, source in sorted(sources.items())
        ), file=sys.stderr, flush=True)
    if env != os.environ:
        env[marker] = str(os.getpid())
        os.execve(sys.executable, [sys.executable, *original[1:]], env)
