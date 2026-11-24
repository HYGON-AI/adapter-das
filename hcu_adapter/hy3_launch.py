# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the HY3 launch environment before framework bridge installation."""

import argparse
import json
import os
from pathlib import Path
import sys

from . import hcu_config, hcu_env
from .glm53_env_rules import resolve_environment
from .glm53_env_rules import LAUNCH_ENV_KEYS as COMMON_ENV_KEYS
from .hy3_profiles import LAUNCH_ENV_KEYS, PROFILES, TP8_PROFILES, preset_env


def checkpoint_family(path):
    model = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if model.get("model_type") != "hy_v3" or "HYV3ForCausalLM" not in model.get("architectures", []):
        return None
    quant = model.get("compression_config") or model.get("quantization_config") or {}
    groups = quant.get("config_groups") or {}
    if quant.get("quant_method") != "compressed-tensors" or not groups:
        return None
    kinds = set()
    for group in groups.values():
        w, a = group.get("weights") or {}, group.get("input_activations") or {}
        if (w.get("num_bits"), w.get("strategy"), a.get("num_bits"),
                a.get("strategy"), a.get("dynamic")) != (8, "channel", 8, "token", True):
            return None
        if w.get("type") != a.get("type") or w.get("type") not in ("int", "float"):
            return None
        kinds.add(w["type"])
    return {"int": "int8", "float": "fp8"}[kinds.pop()] if len(kinds) == 1 else None


def select_profile(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--dp-size", default="1")
    for name in ("nnodes", "pp-size", "ep-size", "attn-cp-size", "attn-dp-size", "dcp-size", "moe-dense-tp-size"):
        parser.add_argument("--" + name, default="1")
    for name in ("tp-size", "moe-a2a-backend", "deepep-mode", "kv-cache-dtype",
                 "attention-backend", "speculative-algorithm", "quantization", "disaggregation-mode"):
        parser.add_argument("--" + name)
    for name in ("enable-dp-attention", "enable-dp-lm-head", "enable-prefill-cp", "enable-hisparse",
                 "disable-radix-cache"):
        parser.add_argument("--" + name, action="store_true")
    args, _ = parser.parse_known_args(argv)
    if not args.model_path or not (args.model_path / "config.json").is_file():
        return None
    family = checkpoint_family(args.model_path)
    if family is None:
        return None
    if (any(getattr(args, name) != "1" for name in (
            "nnodes", "pp_size", "ep_size", "attn_cp_size", "attn_dp_size", "dcp_size", "moe_dense_tp_size"))
            or args.tp_size != "8"
            or args.enable_prefill_cp or args.enable_hisparse or args.disaggregation_mode
            or args.attention_backend != "fa3"):
        return None
    if (args.dp_size == "8" and args.enable_dp_attention and args.enable_dp_lm_head
            and args.moe_a2a_backend == "deepep" and args.deepep_mode == "auto"
            and args.speculative_algorithm == "NEXTN"
            and args.quantization in ((None, "slimquant_marlin") if family == "int8" else (None,))):
        filename = PROFILES.get((family, args.kv_cache_dtype))
    elif (args.dp_size == "1" and not args.enable_dp_attention and not args.enable_dp_lm_head
            and args.moe_a2a_backend is None and args.deepep_mode is None
            and args.quantization is None
            and ((family == "int8" and args.speculative_algorithm is None)
                 or (family == "fp8" and args.speculative_algorithm == "NEXTN" and args.disable_radix_cache))):
        filename = TP8_PROFILES.get((family, args.kv_cache_dtype))
    else:
        return None
    return (family, filename) if filename else None


def apply_official_defaults(argv, original):
    marker = "_HCU_HY3_REEXEC_PID"
    if os.environ.pop(marker, None) == str(os.getpid()):
        return True
    if "--help" in argv or "-h" in argv:
        return False
    preset, _ = hcu_env.resolve("HCU_CONFIG")
    selected = select_profile(argv)
    if preset.startswith("builtin:hy3-") and (
            selected is None or preset != "builtin:" + selected[1]):
        raise SystemExit(f"{preset} does not match this HY3 checkpoint or launch configuration")
    if selected is None:
        return False
    family, filename = selected
    config = hcu_config.load()
    settings = config["hosts"].get("sglang", {})
    configured = {**config["overrides"], **settings.get("env", {})}
    plugins = os.environ.get("SGLANG_PLUGINS", configured.get("SGLANG_PLUGINS"))
    if plugins is not None and "hcu" not in {p.strip() for p in str(plugins).split(",")}:
        if preset.startswith("builtin:hy3-"):
            raise SystemExit(f"{preset} requires the HCU plugin")
        return False
    automatic = not preset and config["path"] is None
    if automatic:
        configured.update(preset_env(filename))
    inherited = os.environ.copy()
    legacy = "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD"
    canonical = "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD"
    if legacy in inherited and canonical not in inherited:
        inherited[canonical] = inherited[legacy]
    env, sources = resolve_environment(None, None, inherited, configured, LAUNCH_ENV_KEYS | COMMON_ENV_KEYS)
    if family == "int8" and env.get("SGLANG_USE_FP8_W8A8_MOE", "0").lower() in ("1", "true"):
        raise SystemExit("HY3 INT8 cannot use SGLANG_USE_FP8_W8A8_MOE=1")
    if automatic:
        env["HCU_CONFIG"] = "builtin:" + filename
        print(f"HCU SGLang config: builtin:{filename} (auto)", file=sys.stderr, flush=True)
    if sources:
        print("HCU HY3 env: " + " ".join(
            f"{key}={env[key]}({source})" for key, source in sorted(sources.items())
        ), file=sys.stderr, flush=True)
    if env != os.environ:
        env[marker] = str(os.getpid())
        os.execve(sys.executable, [sys.executable, *original[1:]], env)
    return True
