# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

from hcu_adapter import bootstrap, hcu_config, hy3_launch, sglang_launch
from hcu_adapter.glm53_env_rules import LAUNCH_ENV_KEYS as COMMON_KEYS
from hcu_adapter.hy3_profiles import LAUNCH_ENV_KEYS, PROFILES, TP8_PROFILES, preset_env


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in LAUNCH_ENV_KEYS | COMMON_KEYS | {"HCU_CONFIG", "_HCU_HY3_REEXEC_PID", "SGL_CHUNKED_PREFIX_CACHE_THRESHOLD"}:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(hcu_config, "_STATE", {"loaded": False, "path": None, "ops": {}, "overrides": {}, "hosts": {}})


def checkpoint(tmp_path, kind):
    path = tmp_path / kind
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "hy_v3", "architectures": ["HYV3ForCausalLM"],
        "compression_config": {"quant_method": "compressed-tensors", "config_groups": {
            "group_0": {"weights": {"num_bits": 8, "strategy": "channel", "type": kind},
                        "input_activations": {"num_bits": 8, "strategy": "token", "type": kind, "dynamic": True}}
        }}
    }), encoding="utf-8")
    return path


def argv(path, kv="fp8_e4m3"):
    return ["--model-path", str(path), "--tp-size", "8", "--dp-size", "8",
            "--moe-a2a-backend", "deepep", "--deepep-mode", "auto",
            "--enable-dp-attention", "--enable-dp-lm-head", "--attention-backend", "fa3",
            "--speculative-algorithm", "NEXTN", "--kv-cache-dtype", kv]


@pytest.mark.parametrize("family,kv", list(PROFILES))
def test_three_profiles_load_and_reexec_with_original_arguments(tmp_path, monkeypatch, capsys, family, kv):
    args = argv(checkpoint(tmp_path, "float" if family == "fp8" else "int"), kv)
    if family == "int8":
        args += ["--quantization=slimquant_marlin"]
    filename = PROFILES[(family, kv)]
    calls = []
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    original = [sys.executable, "-u", "-m", "sglang.launch_server", *args]
    assert hy3_launch.apply_official_defaults(args, original)
    executable, command, env = calls.pop()
    assert executable == sys.executable and command == original
    assert env["HCU_CONFIG"] == "builtin:" + filename
    assert preset_env(filename).items() <= env.items()
    assert env["SGLANG_USE_FP8_W8A8_MOE"] == ("1" if family == "fp8" else "0")
    assert env["SGLANG_USE_FUSED_RMS_ROTARY"] == ("0" if kv == "fp8_e5m2" else "1")
    monkeypatch.setenv("HCU_CONFIG", env["HCU_CONFIG"])
    assert hcu_config.load(force=True)["hosts"]["sglang"]["env"] == preset_env(filename)
    assert "(config)" in capsys.readouterr().err


def test_explicit_values_and_legacy_alias_override_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "5")
    monkeypatch.setenv("SGLANG_USE_FUSED_RMS_ROTARY", "0")
    monkeypatch.setenv("SGL_CHUNKED_PREFIX_CACHE_THRESHOLD", "512")
    calls = []
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    args = argv(checkpoint(tmp_path, "int"))
    hy3_launch.apply_official_defaults(args, [sys.executable, "-m", "sglang.launch_server", *args])
    env = calls[0][2]
    assert env["GPU_MAX_HW_QUEUES"] == "5"
    assert env["SGLANG_USE_FUSED_RMS_ROTARY"] == "0"
    assert env["SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD"] == "512"


def test_custom_file_precedes_automatic_preset_and_explicit_env_wins(tmp_path, monkeypatch):
    config = tmp_path / "custom.json"
    config.write_text(json.dumps({"version": 1, "ops": {"flash_attention": "triton"},
        "hosts": {"sglang": {"env": {"SGLANG_USE_FUSED_RMS_ROTARY": "0", "GPU_MAX_HW_QUEUES": "7"}}}}))
    monkeypatch.setenv("HCU_CONFIG", str(config))
    monkeypatch.setenv("GPU_MAX_HW_QUEUES", "4")
    calls = []
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    args = argv(checkpoint(tmp_path, "int"))
    hy3_launch.apply_official_defaults(args, [sys.executable, "-m", "sglang.launch_server", *args])
    env = calls[0][2]
    assert env["HCU_CONFIG"] == str(config)
    assert env["SGLANG_USE_FUSED_RMS_ROTARY"] == "0"
    assert env["GPU_MAX_HW_QUEUES"] == "4"
    assert env["HCU_FA_IMPL"] == "triton"
    assert "SGLANG_USE_DEEPGEMM_MOE" not in env


@pytest.mark.parametrize("extra", [
    ["--tp-size=4"], ["--dp-size=4"], ["--nnodes=2"], ["--ep-size=8"],
    ["--attn-cp-size=2"], ["--attn-dp-size=2"], ["--enable-hisparse"],
    ["--disaggregation-mode=decode"], ["--attention-backend=triton"],
    ["--deepep-mode=normal"], ["--quantization=compressed-tensors"],
    ["--kv-cache-dtype=bf16"], ["--speculative-algorithm=EAGLE"],
])
def test_unrecorded_topology_keeps_existing_behavior(tmp_path, extra):
    assert hy3_launch.select_profile(argv(checkpoint(tmp_path, "int")) + extra) is None


def test_other_model_and_block_quantization_do_not_match(tmp_path):
    path = checkpoint(tmp_path, "int")
    config = json.loads((path / "config.json").read_text())
    config["compression_config"]["config_groups"]["group_0"]["weights"]["strategy"] = "group"
    (path / "config.json").write_text(json.dumps(config))
    assert hy3_launch.select_profile(argv(path)) is None
    config["model_type"] = "hy_v4"
    (path / "config.json").write_text(json.dumps(config))
    assert hy3_launch.select_profile(argv(path)) is None


def test_disabled_plugin_and_wrong_builtin_are_not_silently_accepted(tmp_path, monkeypatch):
    args = argv(checkpoint(tmp_path, "int"))
    monkeypatch.setenv("SGLANG_PLUGINS", "other")
    assert not hy3_launch.apply_official_defaults(args, [])
    monkeypatch.setenv("HCU_CONFIG", "builtin:hy3-fp8-tp8-dp8-e4m3.yaml")
    with pytest.raises(SystemExit, match="does not match"):
        hy3_launch.apply_official_defaults(args, [])
    monkeypatch.setenv("HCU_CONFIG", "builtin:hy3-int8-tp8-dp8-e4m3.yaml")
    with pytest.raises(SystemExit, match="requires the HCU plugin"):
        hy3_launch.apply_official_defaults(args, [])


def test_int8_rejects_explicit_fp8_expert_computation(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_USE_FP8_W8A8_MOE", "1")
    with pytest.raises(SystemExit, match="INT8 cannot use"):
        hy3_launch.apply_official_defaults(argv(checkpoint(tmp_path, "int")), [])


@pytest.mark.parametrize("module", ["sglang.launch_server", "hcu_sglang", "hcu_sglang.launch_server", "sglang-cli"])
def test_all_module_entries_resolve_before_device_and_bridge_setup(tmp_path, monkeypatch, module):
    from hcu_adapter import sglang_spawn_device

    args = argv(checkpoint(tmp_path, "int"))
    monkeypatch.setattr(sys, "argv", ["sglang", "serve", *args] if module == "sglang-cli" else ["-m", *args])
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "sglang", "serve", *args]
                        if module == "sglang-cli" else [sys.executable, "-u", "-m", module, *args])
    monkeypatch.setattr(bootstrap, "_installed_pid", None)
    monkeypatch.setattr(bootstrap, "_is_hip", lambda: True)
    monkeypatch.setattr(bootstrap, "_available_hosts", lambda: {"sglang"})
    monkeypatch.delenv("_HCU_ADAPTER_ACTIVE_HOST", raising=False)
    events = []
    monkeypatch.setattr(sglang_launch, "apply_official_defaults", lambda *a: events.append("environment"))
    monkeypatch.setattr(sglang_spawn_device, "select_inherited_device", lambda: events.append("device"))
    class Bridge:
        def install(self):
            events.append("bridge")
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: Bridge())
    bootstrap.install()
    assert events == ["environment", "device", "bridge", "bridge"]


def test_real_reexec_applies_environment_once_and_keeps_pid(tmp_path):
    args = argv(checkpoint(tmp_path, "int"))
    script = tmp_path / "probe.py"
    counter = tmp_path / "runs.json"
    script.write_text("\n".join([
        "import json,os,sys", "from pathlib import Path", f"sys.path.insert(0, {str(ROOT)!r})",
        f"sys.path.extend({[p for p in sys.path if 'site-packages' in p]!r})",
        "from hcu_adapter.sglang_launch import apply_official_defaults", f"p=Path({str(counter)!r})",
        "runs=json.loads(p.read_text()) if p.exists() else []", "runs.append(os.getpid())",
        "p.write_text(json.dumps(runs))", "assert len(runs)<=2",
        "apply_official_defaults(sys.argv[1:], sys.orig_argv)",
        "assert '_HCU_HY3_REEXEC_PID' not in os.environ",
        "assert os.environ['HCU_FA_IMPL']=='vendor'",
        "assert os.environ['GLIBC_TUNABLES']=='glibc.rtld.optional_static_tls=0x40000'",
        "print(json.dumps({'pid':os.getpid(),'queues':os.environ['GPU_MAX_HW_QUEUES']}))"
    ]), encoding="utf-8")
    result = subprocess.run([sys.executable, "-S", str(script), *args], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    runs = json.loads(counter.read_text())
    if os.name == "posix":
        assert len(runs) == 2 and runs[0] == runs[1]
    else:
        assert len(runs) == 2
    assert json.loads(result.stdout)["queues"] == "3"


def test_ui_presets_match_packaged_yaml():
    html = (ROOT / "tools/hcu_config_ui.html").read_text(encoding="utf-8")
    presets = json.loads(re.search(r"const HY3_PRESETS = (.*?);\nlet hy3Environment", html, re.S)[1])
    assert presets == {name: preset_env(name) for name in [*PROFILES.values(), *TP8_PROFILES.values()]}


@pytest.mark.parametrize("family,kv", list(TP8_PROFILES))
def test_tp8_examples_select_their_own_environment_without_rewriting_cli(tmp_path, monkeypatch, family, kv):
    filename = TP8_PROFILES[(family, kv)]
    text = (ROOT / "examples/hy3" / filename.replace(".yaml", ".sh")).read_text()
    command = shlex.split(text.replace("\\\n", ""), comments=True)
    assert command[:3] == ["python3", "-m", "sglang.launch_server"]
    args = command[3:]
    args[args.index("--model-path") + 1] = str(checkpoint(tmp_path, "float" if family == "fp8" else "int"))
    assert hy3_launch.select_profile(args) == (family, filename)
    original = [sys.executable, "-m", "sglang.launch_server", *args]
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
    assert hy3_launch.apply_official_defaults(args, original)
    _, actual_command, env = calls.pop()
    assert actual_command == original
    assert preset_env(filename).items() <= env.items()
    assert env["HCU_CONFIG"] == "builtin:" + filename
    assert "SGLANG_USE_DEEPGEMM_MOE" not in env
    assert env["HCU_FA_IMPL"] == "vendor"
    assert env["SGLANG_KV_LAYOUT_HCU_FA"] == "1"
    if family == "int8":
        assert env["SGLANG_ROCM_USE_AITER_MOE"] == "false"
        assert env["SGLANG_USE_FUSED_RMS_ROTARY"] == "0"
        assert env["W8A8_SUPPORT_METHODS"] == "2"
        assert "--speculative-algorithm" not in actual_command
        assert ("--dtype" in actual_command) == (kv == "fp8_e4m3")
    else:
        assert env["USE_DCU_CUSTOM_ALLREDUCE"] == "1"
        assert env["SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH"] == "1"
        assert "--disable-radix-cache" in actual_command


@pytest.mark.parametrize("family,extra", [
    ("int", ["--dp-size=2"]), ("int", ["--enable-dp-attention"]),
    ("int", ["--speculative-algorithm=NEXTN"]), ("int", ["--moe-a2a-backend=deepep"]),
    ("float", []), ("float", ["--speculative-algorithm=NEXTN"]),
    ("float", ["--speculative-algorithm=EAGLE", "--disable-radix-cache"]),
])
def test_tp8_does_not_apply_to_other_parallel_or_draft_modes(tmp_path, family, extra):
    args = ["--model-path", str(checkpoint(tmp_path, family)), "--tp-size=8",
            "--attention-backend=fa3", "--kv-cache-dtype=fp8_e4m3", *extra]
    assert hy3_launch.select_profile(args) is None


def test_tp8_explicit_environment_and_custom_config_keep_priority(tmp_path, monkeypatch):
    config = tmp_path / "custom.json"
    config.write_text(json.dumps({"version": 1, "hosts": {"sglang": {"env": {
        "W8A8_SUPPORT_METHODS": "3", "SGLANG_USE_FUSED_RMS_ROTARY": "1"}}}}))
    monkeypatch.setenv("HCU_CONFIG", str(config))
    monkeypatch.setenv("W8A8_SUPPORT_METHODS", "1")
    monkeypatch.setenv("HCU_FA_IMPL", "triton")
    monkeypatch.setenv("SGLANG_KV_LAYOUT_HCU_FA", "0")
    args = ["--model-path", str(checkpoint(tmp_path, "int")), "--tp-size=8",
            "--attention-backend=fa3", "--kv-cache-dtype=fp8_e4m3"]
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
    assert hy3_launch.apply_official_defaults(args, [sys.executable, "-m", "sglang.launch_server", *args])
    env = calls[0][2]
    assert env["W8A8_SUPPORT_METHODS"] == "1"
    assert env["HCU_FA_IMPL"] == "triton"
    assert env["SGLANG_KV_LAYOUT_HCU_FA"] == "0"
    assert env["SGLANG_USE_FUSED_RMS_ROTARY"] == "1"
    assert "SGLANG_ROCM_USE_AITER_MOE" not in env


@pytest.mark.parametrize("builtin", [False, True])
def test_existing_glm53_selection_is_preserved(tmp_path, monkeypatch, builtin):
    from hcu_adapter import glm53_launch

    model = tmp_path / "glm"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "glm_moe_dsa", "hidden_size": 6144,
        "num_hidden_layers": 78, "n_routed_experts": 256,
        "quantization_config": {"quant_method": "compressed-tensors", "config_groups": {
            "experts": {"targets": ["re:.*experts.*"],
                        "weights": {"num_bits": 4, "type": "int", "strategy": "channel"},
                        "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "dynamic": True}}
        }}
    }))
    args = ["--model-path", str(model), "--tp-size", "8", "--quantization", "compressed-tensors"]
    if builtin:
        monkeypatch.setenv("HCU_CONFIG", "builtin:glm53-w4a8-ifb-tp8.yaml")
    calls = []
    monkeypatch.setattr(os, "execve", lambda *args: calls.append(args))
    glm53_launch.apply_official_defaults(args, [sys.executable, "-m", "sglang.launch_server", *args])
    expected = calls.pop()
    sglang_launch.apply_official_defaults(args, [sys.executable, "-m", "sglang.launch_server", *args])
    assert calls == [expected]
    env = calls[0][2]
    if builtin:
        assert env["HCU_CONFIG"] == "builtin:glm53-w4a8-ifb-tp8.yaml"
    else:
        assert "HCU_CONFIG" not in env
    assert env["HCU_W4A8_MOE"] == "native"
    assert "SGLANG_USE_FUSED_RMS_ROTARY" not in env
