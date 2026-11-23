# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Select a GLM-5.3 launch profile from the checkpoint and server arguments."""

import argparse
import json
import os
import sys
from pathlib import Path

from . import hcu_config, hcu_env
from . import glm53_fp8_decode as dp32
from .glm53_fp8_decode import _channel_fp8, launch_environment
from .glm53_profiles import PROFILES, Profile, preset_env

PARALLEL_SIZES = (
    "nnodes", "tp-size", "pp-size", "ep-size", "dp-size", "attn-dp-size",
    "attn-cp-size", "dcp-size",
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


def parse_options(argv, *, optional=False):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-path", required=not optional, type=Path)
    parser.add_argument("--disaggregation-ib-device")
    for option in PARALLEL_SIZES:
        parser.add_argument("--" + option, default="1")
    for option in ("quantization", "disaggregation-mode", "disaggregation-transfer-backend", "moe-a2a-backend", "deepep-mode"):
        parser.add_argument("--" + option)
    for option in ("enable-prefill-cp", "enable-dp-attention", "enable-hisparse"):
        parser.add_argument("--" + option, action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def matching_profile(args, family):
    if family is None or args.quantization not in (None, "compressed-tensors"):
        return None
    settings = vars(args).copy()
    settings.pop("model_path")
    devices = settings.pop("disaggregation_ib_device")
    # These checkpoints declare compressed-tensors in config.json. An explicit
    # equivalent CLI flag must select the same preset as the inferred value.
    settings["quantization"] = "compressed-tensors" if family == "w4a8" else None
    topology = {key: value for key, value in settings.items() if value not in (None, False, "1")}
    for name, profile in PROFILES.items():
        if name.startswith(f"glm53_{family}_") and profile.topology == topology:
            return name, profile, devices
    dp32_topology = PROFILES["glm53_fp8_decode_dp8"].topology.copy()
    dp32_topology.update(nnodes="8", tp_size="32", ep_size="32", attn_dp_size="32")
    if family == "fp8" and {k: v for k, v in topology.items() if k != "enable_hisparse"} == dp32_topology:
        return "glm53_fp8_decode_dp32", Profile("glm53-fp8-decode.yaml", dp32.SITE_KEYS, dp32.DERIVED_KEYS, dp32_topology, offline=True), devices
    return None


def select_profile(argv):
    args = parse_options(argv)
    selected = matching_profile(args, checkpoint_family(args.model_path))
    parser = argparse.ArgumentParser()
    if selected is None:
        parser.error("unsupported GLM-5.3 checkpoint, role or parallel configuration")
    if args.disaggregation_mode and not args.disaggregation_ib_device:
        parser.error("P/D profiles require --disaggregation-ib-device for this node")
    return selected


def profile_environment(selection):
    name, profile, devices = selection
    config = hcu_config.load()
    configured = {**config["overrides"], **config["hosts"].get("sglang", {}).get(name, {})}
    env, sources = launch_environment(
        os.environ, configured, devices, defaults=profile.defaults,
        site=profile.site, derived=profile.derived,
    )
    if name.startswith("glm53_w4a8_") and env["HCU_W4A8_MOE"] != "native":
        raise SystemExit(f"{name} requires HCU_W4A8_MOE=native")
    print(name + ": " + " ".join(
        f"{k}={env[k]}({source})" for k, source in sorted(sources.items())
    ), file=sys.stderr, flush=True)
    return env


def apply_official_defaults(argv, original):
    # exec keeps the PID and makes loader and device defaults visible from
    # process creation. Consume the marker before SGLang spawns any workers.
    marker = "_HCU_GLM53_REEXEC_PID"
    if os.environ.pop(marker, None) == str(os.getpid()):
        return
    if "--help" in argv or "-h" in argv:
        return
    from .glm53_env_rules import resolve_environment
    from .glm53_profiles import LAUNCH_ENV_KEYS

    args = parse_options(argv, optional=True)
    config = hcu_config.load()
    settings = config["hosts"].get("sglang", {})
    configured = {**config["overrides"], **settings.get("env", {})}
    family = None
    plugins = os.environ.get("SGLANG_PLUGINS", configured.get("SGLANG_PLUGINS"))
    hcu_allowed = plugins is None or "hcu" in {p.strip() for p in str(plugins).split(",")}
    if hcu_allowed and args.model_path and (args.model_path / "config.json").is_file():
        family = checkpoint_family(args.model_path)
    preset, _ = hcu_env.resolve("HCU_CONFIG")
    selected = matching_profile(args, family)
    auto_preset = None
    if selected is not None:
        name, profile, _ = selected
        if not preset and config["path"] is None:
            auto_preset = "builtin:" + profile.filename
            configured.update(preset_env(profile.filename))
        configured.update(settings.get(name, {}))
    if preset.startswith("builtin:glm53-"):
        expected_family = preset.split("-", 2)[1]
        if family != expected_family:
            raise SystemExit(f"{preset} requires a local GLM-5.3 {expected_family} checkpoint and HCU plugin")
    env, sources = resolve_environment(args, family, os.environ, configured, LAUNCH_ENV_KEYS)
    if family == "w4a8" and env["HCU_W4A8_MOE"] != "native":
        raise SystemExit("GLM-5.3 W4A8 requires native MoE; check HCU_W4A8_MOE")
    if auto_preset is not None:
        # Workers and the restarted interpreter discover the identical file.
        env["HCU_CONFIG"] = auto_preset
        print(f"HCU SGLang config: {auto_preset} (auto)", file=sys.stderr, flush=True)
    if sources:
        print("HCU SGLang env: " + " ".join(
            f"{key}={env[key]}({source})" for key, source in sorted(sources.items())
        ), file=sys.stderr, flush=True)
    if env != os.environ:
        env[marker] = str(os.getpid())
        os.execve(sys.executable, [sys.executable, *original[1:]], env)


def main():
    argv = sys.argv[1:]
    env = profile_environment(select_profile(argv))
    # Preserve interpreter options such as -u on the new service process.
    original = getattr(sys, "orig_argv", ())
    interpreter_args = original[1:original.index("-m")] if "-m" in original else []
    if not original and getattr(sys.stdout, "write_through", False):
        interpreter_args = ["-u"]
    os.execve(sys.executable, [sys.executable, *interpreter_args, "-m", "hcu_sglang", *argv], env)


if __name__ == "__main__":
    main()
