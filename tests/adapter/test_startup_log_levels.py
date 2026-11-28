# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Startup log levels: one summary per service, a default-quiet install chain,
a guard on new WARNING sites (governance §3b-8).

Run: python3 -m pytest -q tests/adapter/test_startup_log_levels.py
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import pytest

import hcu_adapter.infra.bootstrap as bootstrap
from hcu_adapter.infra import hcu_env, registry
from hcu_adapter.infra.bootstrap import ACTIVE_HOST_ENV

ADAPTER = "hcu.adapter"
ROOT = Path(__file__).resolve().parents[2]
VENDORED = {"fla_hcu"}  # third-party FLA sources keep their own logging

# WARNING lines outside every except handler split in two. Key of a site:
# "<file> <function> <first 40 characters of the message>".
#
# SERVICE_STATE: service-level state (an effective route, a gate, an environment
# fact). Each carries extra=SERVICE_STATE, so the top-level process prints it
# once and a spawn child does not repeat it.
SERVICE_STATE = {
    "attn_backend_gate.py install HCU adapter: CUDA-only attention backend",
    "bmm_fp8_bridge.py install HCU adapter: fp8_utils.bmm_fp8 bridged t",
    "bridges/sglang_aiter_preshuffle.py install HCU adapter: DSA aiter-preshuffle gate f",
    "bridges/sglang_cp.py install [HCU mhcunfuse] SGLANG_OPT_FUSE_MHC_POST",
    "bridges/sglang_fa_vendor.py install HCU adapter: HCU_FA_IMPL=%s -- fa3 atten",
    "bridges/sglang_fa_vendor.py install HCU adapter: vendor_fa disabled; using v",
    "bridges/sglang_family_overrides.py install HCU adapter: keeping page_size=%d for DS",
    "bridges/sglang_family_overrides.py install HCU adapter: page_size %d not a multiple",
    "bridges/sglang_multimem_gate.py install HCU adapter: multimem all-gather probe d",
    "bridges/sglang_native_ops_gate.py install HCU adapter: native sgl_kernel ops prese",
    "bridges/sglang_native_ops_gate.py install HCU adapter: native-ops serve gate armed",
    "bridges/sglang_spec_kv.py install HCU adapter: absent idle verify KV metad",
    "features/aiter/aiter_jit_cache.py report HCU aiter jit cache: + %s",
    "features/aiter/aiter_jit_cache.py report HCU aiter jit cache: wheel=%s prebuilt_s",
    "features/aiter/ar_norm_aiter.py _install_dispatch [HCU ar+norm] HCU_AR_NORM_IMPL=stock -- ",
    "features/aiter/ar_norm_aiter.py _install_dispatch [HCU ar+norm] upstream candidate gate ab",
    "features/dsa/fused_a_gemm_gate.py install HCU adapter: fused QKV-A GEMM min-latenc",
    "features/gdn/gdn_aiter.py _install_target_verify_aiter HCU adapter: GDN verify kept upstream by",
    "features/gdn/gdn_aiter.py install HCU adapter: HCU_LINEAR_ATTN_GDN=triton ",
    "features/gdn/mamba_state_scatter.py install HCU adapter: Mamba state scatter kept up",
    "features/jit/small_copy_jit.py install HCU adapter: small_copy uses standard Tr",
    "infra/kill_switch.py log_state HCU adapter: HCU_DISABLE_OPS has unknown",
    "infra/kill_switch.py log_state HCU adapter: kill switches ACTIVE: %s",
    "infra/nccl_topo_guard.py install HCU adapter: NCCL_TOPO_FILE cleared (%s ",
    "infra/platform_seam.py install HCU adapter: platform semantics=%r (lega",
    "infra/triton_libdev.py install HCU adapter: triton libdevice.%s has no ",
    "infra/tvmffi_warmup.py _build_with_clean_env HCU adapter: tvm-ffi dlpack cache cold -",
    "infra/watchdog_cold.py _log_bump HCU adapter: cold JIT caches detected ->",
    "moe_dualstream_gate.py install HCU adapter: DSV2 capture-mode dual-stre",
    "prefill_cp.py validate_prefill_cp_platform_hcu HCU prefill-CP bridge: platform gate ope",
}

# PER_PROCESS: every process reports its own: the bootstrap line and the
# once-per-service summary lines, refusals, fallbacks, failures, "not installed",
# a cold cache that stayed cold, a per-process selection, an opt-in debug probe.
# Never marked, and never an "installed" confirmation (those are INFO and need
# HCU_ADAPTER_LOG=info).
PER_PROCESS = {
    "_base.py __mro_entries__ HCU adapter: '%s' missing from %s -- sub",
    "bridges/sglang_aiter_ar.py install HCU adapter: aiter custom all-reduce ref",
    "bridges/sglang_deep_ep.py install dropped unmapped kwargs: %s",
    "bridges/sglang_deep_ep.py install x_global_scale dropped on bf16 dispatch ",
    "bridges/sglang_deep_gemm.py install HCU DeepGEMM bridge: wrapper re-import s",
    "bridges/sglang_diffusion_numerics.py warn_unsupported_arches HCU adapter: skipping SGLang diffusion F",
    "bridges/sglang_fa_vendor.py install FA-BRIDGE kvcache: nonNone=%s causal=%s",
    "bridges/sglang_layer_audit.py install HCU adapter: layer audit installed -> %s",
    "bridges/sglang_native_ops_gate.py install %s",
    "bridges/sglang_spawn_device.py select_inherited_device HCU adapter: selected scheduler GPU %s b",
    "bridges/vllm.py _log_post_install_bindings HCU adapter install-done: %s <- %s",
    "bridges/vllm_input_quant_int8_lightop.py _flush HCU LightOp per-token INT8 counters: %s",
    "bridges/vllm_input_quant_int8_lightop.py install HCU LightOp per-token INT8: skipped (sca",
    "bridges/vllm_input_quant_int8_lightop.py install HCU adapter: LightOp per-token INT8 quan",
    "bridges/vllm_input_quant_lightop.py _flush HCU LightOp per-token FP8 amax diag: cal",
    "bridges/vllm_input_quant_lightop.py _flush HCU LightOp per-token FP8 counters: %s",
    "bridges/vllm_input_quant_lightop.py install HCU LightOp per-token FP8: skipped (%s m",
    "bridges/vllm_input_quant_lightop.py install HCU LightOp per-token FP8: skipped (%s.Q",
    "bridges/vllm_input_quant_lightop.py install HCU LightOp per-token FP8: skipped (Grou",
    "bridges/vllm_input_quant_lightop.py install HCU LightOp per-token FP8: skipped (forw",
    "bridges/vllm_input_quant_lightop.py install HCU adapter: LightOp per-token FP8 quant",
    "features/aiter/aiter_moe.py _dbg HCU aiter moe debug: %s",
    "features/aiter/aiter_moe.py try_aiter_fused_moe HCU aiter moe: no validated ASM/moe_c co",
    "features/aiter/ar_norm_aiter.py _install_dispatch [HCU ar+norm] fused kernel declined; usi",
    "features/aiter/quant_aiter.py install [HCU quant] aiter %s failed -- fell back",
    "features/ct_w4a8/ct_w4a8_moe.py _make_patched_init HCU adapter: CT W4A8 -> W4A16 route: dro",
    "features/ct_w4a8/moe_c_w4a16.py _dbg HCU moe_c w4a16: %s",
    "features/ct_w4a8/moe_c_w4a16.py try_moec_w4a16 HCU moe_c w4a16: shape miss %s (log-once",
    "features/ct_w4a8/w4a16_moe.py _dbg HCU w4a16 moe: %s",
    "features/dsa/dsa_mqa_logits_emu.py _warn_once msg",
    "features/dsv4/dsv4_silu_fused_quant.py _hcu_silu_fused_quant HCU silu fused-quant route MISS (once): ",
    "features/gdn/gdn_cutedsl_fallback.py install HCU adapter: gdn CuTeDSL fallback skippe",
    "hook_aliases.py install HCU hook alias optimization: native help",
    "hook_aliases.py install HCU hook alias optimization: source chan",
    "hook_aliases.py install HCU hook alias optimization: unsupported",
    "infra/bootstrap.py install HCU adapter bootstrap: host=%s source=%s",
    "infra/hcu_env.py log_summary 'HCU adapter env: ' + ' '.join(parts)",
    "infra/hcu_env.py log_summary HCU adapter dump: %s",
    "infra/hcu_env.py log_summary HCU adapter env: all defaults (no explic",
    "infra/hcu_env.py log_summary cfg_line",
    "infra/mapped_host_dtk.py _device_ptr_for_host_tensor PLE hipHostGetDevicePointer failed rc=%s",
    "infra/platform_seam.py install HCU adapter: CUDA face unavailable at in",
    "infra/registry.py log_summary HCU adapter runtime: %-18s <- %-40s [%s/",
    "infra/registry.py log_summary HCU adapter runtime: catalog ops on upst",
    "infra/registry.py log_summary HCU adapter runtime: no ops installed (p",
    "infra/tier_policy.py pick [HCU tier] %s: explicit %s is outside th",
    "infra/tvmffi_warmup.py install HCU adapter: tvm-ffi dlpack cache still ",
    "infra/watchdog_cold.py install HCU adapter: cold-JIT watchdog floor not",
    "kernels/mhc_aiter.py load_provider AITER TileLang mHC enabled but pre_big_f",
    "kernels/mhc_aiter.py load_provider AITER mHC capability %s absent; this rou",
    "kernels/mhc_mega.py _count_prepare mHC mega Fn has no mutation version; rep",
    "kernels/mla_decode_rope_triton.py qualify_inverse_rope HCU inverse RoPE preflight output differ",
    "kernels/mla_native_plan_triton.py qualify_native_plan HCU FlashMLA plan preflight found an inc",
    "kernels/mla_native_plan_triton.py qualify_native_plan HCU FlashMLA plan preflight found differ",
    "kernels/mla_selectors_triton.py _latch 'triton fast launch fallback: ' + msg",
    "prefill_cp.py _patch_zigzag_module CPDBG[%s r%s] in(%s) out(%s)",
    "prefill_cp.py _patch_zigzag_module HCU prefill-CP bridge: CPDBG instrumenta",
    "prefill_cp.py _wrap_forward_cuda HCU prefill-CP bridge: CP chunked extend",
}


def _level_for(monkeypatch, raw):
    log = logging.getLogger(ADAPTER)
    monkeypatch.delenv("HCU_ADAPTER_LOG", raising=False)
    if raw is not None:
        monkeypatch.setenv("HCU_ADAPTER_LOG", raw)
    bootstrap._apply_adapter_log_level()
    return log


def test_switch_is_registered_with_an_empty_default(monkeypatch):
    monkeypatch.delenv("HCU_ADAPTER_LOG", raising=False)
    assert hcu_env.REGISTRY["HCU_ADAPTER_LOG"]["env"] == "HCU_ADAPTER_LOG"
    assert hcu_env.REGISTRY["HCU_ADAPTER_LOG"]["default"] == ""
    assert hcu_env.resolve("HCU_ADAPTER_LOG", evaluate_auto=False) == ("", "default")


@pytest.mark.parametrize(
    "raw,level",
    [
        (None, logging.WARNING),
        ("", logging.WARNING),
        ("INFO", logging.INFO),
        ("debug", logging.DEBUG),
        ("verbose", logging.WARNING),
    ],
)
def test_adapter_logger_defaults_to_warning(monkeypatch, raw, level):
    assert _level_for(monkeypatch, raw).level == level


def test_install_lines_are_silent_until_requested(monkeypatch, caplog):
    log = _level_for(monkeypatch, None)
    log.info("HCU adapter: installed (sample)")
    assert "installed (sample)" not in caplog.text

    _level_for(monkeypatch, "info")
    log.info("HCU adapter: installed (sample)")
    assert "installed (sample)" in caplog.text


def test_boot_lines_go_to_stderr_with_a_timestamp_until_the_host_logs(monkeypatch, capsys):
    log = logging.getLogger(ADAPTER + ".boot-test")
    log.setLevel(logging.INFO)
    handler = bootstrap._boot_handler()
    log.addHandler(handler)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(logging.getLogger(), "handlers", [])
            log.info("HCU adapter: installed (sample)")
        err = capsys.readouterr().err
        assert re.fullmatch(
            r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] HCU adapter: installed \(sample\)\n", err
        ), err
        # once the host configured logging the record propagates only: no second copy here
        log.info("HCU adapter: installed (sample)")
        assert capsys.readouterr().err == ""
    finally:
        log.removeHandler(handler)


def test_service_state_is_dropped_in_a_spawn_child_unless_forced(monkeypatch, caplog):
    log = _level_for(monkeypatch, None)
    monkeypatch.delenv("HCU_ADAPTER_LOG", raising=False)
    monkeypatch.setattr(bootstrap, "_spawned_child", False)
    with caplog.at_level(logging.WARNING, logger=ADAPTER):
        log.warning("[HCU state] effective route", extra=bootstrap.SERVICE_STATE)
        assert "[HCU state] effective route" in caplog.text
        caplog.clear()

        monkeypatch.setattr(bootstrap, "_spawned_child", True)
        log.warning("[HCU state] effective route", extra=bootstrap.SERVICE_STATE)
        log.warning("HCU adapter: sample bridge not installed", exc_info=ValueError("x"))
        log.error("[HCU state] still an error", extra=bootstrap.SERVICE_STATE)
        assert "[HCU state] effective route" not in caplog.text
        assert "sample bridge not installed" in caplog.text
        assert "[HCU state] still an error" in caplog.text
        caplog.clear()

        monkeypatch.setenv("HCU_ADAPTER_LOG", "info")
        with caplog.at_level(logging.INFO, logger=ADAPTER):
            log.warning("[HCU state] effective route", extra=bootstrap.SERVICE_STATE)
            log.info("HCU adapter: installed (sample)")
        assert "[HCU state] effective route" in caplog.text
        assert "installed (sample)" in caplog.text


def _arm_install(monkeypatch, *, child, raw):
    seen = []

    class Bridge:
        def __init__(self, name):
            self.name = name

        def install(self):
            seen.append(self.name)
            if self.name == "common":
                hcu_env.log_summary()
            elif self.name == "sglang":
                # what the real bridge does at the end of install()
                if hcu_env.summary_due():
                    registry.log_summary()

    modules = {
        "hcu_adapter.bridges.common": Bridge("common"),
        "hcu_adapter.bridges.sglang": Bridge("sglang"),
    }
    monkeypatch.setattr(bootstrap, "_is_hip", lambda: True)
    monkeypatch.setattr(bootstrap, "_available_hosts", lambda: {"sglang"})
    monkeypatch.setattr(bootstrap, "_loaded_module_names", lambda: ())
    monkeypatch.setattr(bootstrap.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(bootstrap, "_installed_pid", None)
    monkeypatch.setattr(bootstrap, "_spawned_child", False)
    monkeypatch.delenv("HCU_ADAPTER_DUMP", raising=False)
    monkeypatch.setenv("HCU_ADAPTER_HOST", "auto")
    monkeypatch.delenv("HCU_ADAPTER_LOG", raising=False)
    if raw is not None:
        monkeypatch.setenv("HCU_ADAPTER_LOG", raw)
    if child:
        monkeypatch.setenv(ACTIVE_HOST_ENV, "sglang")
    else:
        monkeypatch.delenv(ACTIVE_HOST_ENV, raising=False)
    return seen


@pytest.mark.parametrize(
    "child,raw,printed",
    [
        (False, None, True),  # top-level process: the service summary
        (True, None, False),  # spawn child repeats the install, not the summary
        (True, "info", True),  # per-process troubleshooting
    ],
)
def test_summary_is_printed_once_per_service(monkeypatch, caplog, child, raw, printed):
    seen = _arm_install(monkeypatch, child=child, raw=raw)

    with caplog.at_level(logging.WARNING, logger=ADAPTER):
        bootstrap.install()

    assert seen == ["common", "sglang"]
    assert bootstrap.is_spawned_child() is child
    # the per-process bootstrap line stays in every process
    assert f"host=sglang source={'inherited' if child else 'availability'}" in caplog.text
    assert ("HCU adapter env:" in caplog.text) is printed
    assert ("HCU adapter runtime:" in caplog.text) is printed


def test_dump_is_not_gated_by_the_spawn_check(monkeypatch, caplog):
    _arm_install(monkeypatch, child=True, raw=None)
    monkeypatch.setenv("HCU_ADAPTER_DUMP", "1")

    with caplog.at_level(logging.WARNING, logger=ADAPTER):
        bootstrap.install()

    assert "HCU adapter dump:" in caplog.text


def _tree(name):
    return ast.parse((ROOT / "hcu_adapter" / name).read_bytes(), name)


def _calls(tree, attribute):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == attribute:
                yield node


def test_chain_head_no_longer_prints_the_per_op_table():
    assert not list(_calls(_tree("infra/hcu_env.py"), "log_summary"))


def test_sglang_bridge_closes_with_the_per_op_table():
    install = next(
        node for node in _tree("bridges/sglang.py").body
        if isinstance(node, ast.FunctionDef) and node.name == "install"
    )
    gates = [node for node in ast.walk(install) if isinstance(node, ast.If) and ast.unparse(node.test) == "hcu_env.summary_due()"]
    assert len(gates) == 1
    gate = gates[0]
    assert [ast.unparse(call.func) for call in _calls(gate, "log_summary")] == ["registry.log_summary"]
    assert not list(_calls(gate, "register"))  # only the print is gated, never a binding


def _sglang_bridges():
    return sorted((ROOT / "hcu_adapter" / "bridges").glob("sglang*.py"))


def test_sglang_bridges_are_covered():
    assert len(list(_sglang_bridges())) > 10


def test_bridge_prints_are_opt_in():
    """Bare prints skip the logger: in the sglang bridges they only exist behind a debug switch."""
    checked = 0
    for path in _sglang_bridges():
        tree = ast.parse(path.read_bytes(), str(path))
        parent = {}
        for node in ast.walk(tree):
            for kid in ast.iter_child_nodes(node):
                parent[kid] = node
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"):
                continue
            checked += 1
            current, opt_in = node, False
            while current in parent:
                outer = parent[current]
                if isinstance(outer, ast.If) and re.search(r"environ|hcu_env|getenv", ast.unparse(outer.test)):
                    opt_in = True
                if isinstance(outer, ast.ExceptHandler):
                    opt_in = True
                current = outer
            assert opt_in, "print at %s:%d is not behind a debug switch" % (path.name, node.lineno)
    assert checked > 10


def _marked(call):
    for keyword in call.keywords:
        if keyword.arg == "extra" and isinstance(keyword.value, ast.Dict):
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if isinstance(key, ast.Constant) and key.value == "once_per_service":
                    return isinstance(value, ast.Constant) and value.value is True
    return False


def _warning_sites():
    """key -> [marked, ...]: one entry per WARNING call outside every except handler."""
    sites = {}
    for path in sorted((ROOT / "hcu_adapter").rglob("*.py")):
        if VENDORED & set(path.relative_to(ROOT / "hcu_adapter").parts):
            continue
        tree = ast.parse(path.read_bytes(), str(path))
        function, handled = {}, {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = node.name
            elif isinstance(node, ast.Lambda):
                scope = "<lambda>"
            else:
                continue
            for kid in ast.walk(node):
                function.setdefault(kid, scope)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                for statement in handler.body:
                    for kid in ast.walk(statement):
                        handled.setdefault(kid, True)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("warning", "warn")):
                continue
            if node in handled:
                continue
            first = node.args[0] if node.args else (node.keywords[0].value if node.keywords else None)
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                message = first.value
            else:
                message = ast.unparse(first)
            name = path.relative_to(ROOT / "hcu_adapter").as_posix()
            sites.setdefault("%s %s %s" % (name, function.get(node, "<module>"), message.strip()[:40]), []).append(_marked(node))
    return sites


def test_warning_outside_except_handlers_is_classified():
    sites = _warning_sites()
    unexpected = set(sites) - SERVICE_STATE - PER_PROCESS
    assert not unexpected, (
        "new WARNING outside an except handler (governance §4: an install confirmation is "
        "INFO, silent unless HCU_ADAPTER_LOG=info):\n  " + "\n  ".join(sorted(unexpected))
        + "\nDemote it to .info(...) if it only confirms an install; service-level state goes to "
        "SERVICE_STATE with extra=SERVICE_STATE; anything else a process must report by itself "
        "goes to PER_PROCESS."
    )
    stale = (SERVICE_STATE | PER_PROCESS) - set(sites)
    assert not stale, "classified entries without a call site:\n  " + "\n  ".join(sorted(stale))
    overlap = SERVICE_STATE & PER_PROCESS
    assert not overlap, overlap
    unmarked = sorted(key for key in SERVICE_STATE if not all(sites[key]))
    assert not unmarked, "service-level state without extra=SERVICE_STATE:\n  " + "\n  ".join(unmarked)
    stray = sorted(key for key in PER_PROCESS if any(sites[key]))
    assert not stray, "per-process lines must not carry extra=SERVICE_STATE:\n  " + "\n  ".join(stray)
