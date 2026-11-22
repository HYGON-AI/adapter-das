# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""jit_qwen4 / triton_pdl contracts (CPU only, no GPU).

Covers the two v0.5.22 Qwen3.8-Flash-Next boot failures:

  - `elementwise/grouped_gemma_rmsnorm.cuh` structured-binds into a HIP vector
    type -> hipcc "cannot decompose class type ... anonymous union member".
    jit_qwen4 rewrites the three sites to `.x/.y` member reads.
  - `create_grid_dependency_*` has no HCU lowering -> `PassManager::run
    failed` inside `ConvertTritonHCUToLLVM`. triton_pdl neutralises the
    frontend semantic methods (and keeps the asm stripper for wheels that
    lower the ops).
"""

import types

import pytest

from hcu_adapter import jit_qwen4, registry, triton_pdl

# The three CUDA-isms and the observable part of each replacement.
_CASES = (
    (
        "      const auto [x, y] = cast<fp32x2_t>(input_vec[j][i]);",
        ("const fp32x2_t _xy = cast<fp32x2_t>(input_vec[j][i]);",
         "const auto x = _xy.x, y = _xy.y;"),
    ),
    (
        "      const auto [ix, iy] = cast<fp32x2_t>(input_vec[j][i]);",
        ("const fp32x2_t _ixy = cast<fp32x2_t>(input_vec[j][i]);",
         "const auto ix = _ixy.x, iy = _ixy.y;"),
    ),
    (
        "      const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j][i]);",
        ("const fp32x2_t _wxy = cast<fp32x2_t>(weight_vec[j][i]);",
         "const auto wx = _wxy.x, wy = _wxy.y;"),
    ),
)


def _source(tmp_path, body=None):
    root = tmp_path / "bundle"
    target = root / "csrc" / jit_qwen4._RELPATH
    target.parent.mkdir(parents=True)
    target.write_text(
        body if body is not None else "\n".join(n for n, _ in _CASES) + "\n"
    )
    return root, target


def _recorder():
    records = []

    def register(*args, **kwargs):
        records.append((args, kwargs))

    return records, register


def test_needles_are_the_declared_patch_set():
    assert tuple(n for n, _ in _CASES) == tuple(n for n, _ in jit_qwen4._PATCHES)


def test_needles_are_allowlisted():
    """Every needle must be governance-reviewed in TEXT_PATCH_ALLOWLIST."""
    for needle, _ in _CASES:
        assert ("rms_norm", needle) in registry.TEXT_PATCH_ALLOWLIST, needle


def test_patch_rewrites_all_three_sites(tmp_path):
    root, target = _source(tmp_path)
    records, register = _recorder()
    jit_qwen4._apply(root, register)

    text = target.read_text()
    for needle, fragments in _CASES:
        assert needle not in text, needle
        for fragment in fragments:
            assert fragment in text, fragment
    # one allowlist record per needle, all under the rms_norm op
    assert len(records) == len(_CASES)
    for args, kwargs in records:
        assert args[0] == "rms_norm"
        assert kwargs["mechanism"] == registry.MECH_TEXT
        assert kwargs["target"].endswith(jit_qwen4._RELPATH)


def test_patch_is_idempotent(tmp_path):
    root, target = _source(tmp_path)
    records, register = _recorder()
    jit_qwen4._apply(root, register)
    once = target.read_text()
    calls = len(records)
    jit_qwen4._apply(root, register)
    assert target.read_text() == once
    assert len(records) == calls  # no duplicate allowlist records


def test_absent_file_is_skipped(tmp_path):
    root = tmp_path / "bundle"
    (root / "csrc").mkdir(parents=True)
    records, register = _recorder()
    jit_qwen4._apply(root, register)  # must not raise
    assert records == []


def test_missing_needle_raises(tmp_path):
    root, _ = _source(tmp_path, body="// upstream dropped the kernel\n")
    _, register = _recorder()
    with pytest.raises(RuntimeError, match="needle missing"):
        jit_qwen4._apply(root, register)


def test_transform_registered_in_bundle_pipeline():
    from hcu_adapter import jit_sources

    assert "jit_qwen4" in jit_sources._TRANSFORMS


# --- triton_pdl -----------------------------------------------------------

def _fake_semantic():
    calls = []

    class FakeSemantic:
        def grid_dependency_wait(self):
            calls.append("wait")

        def grid_dependency_launch_dependents(self):
            calls.append("launch")

    return types.SimpleNamespace(TritonSemantic=FakeSemantic), FakeSemantic, calls


def test_frontend_noop_replaces_semantic_methods(monkeypatch):
    module, FakeSemantic, calls = _fake_semantic()
    monkeypatch.setattr(triton_pdl, "_semantic_module", lambda: module)

    installed = triton_pdl._install_frontend_noop()
    assert installed == (
        "grid_dependency_wait", "grid_dependency_launch_dependents",
    )
    semantic = FakeSemantic()
    assert semantic.grid_dependency_wait() is None
    assert semantic.grid_dependency_launch_dependents() is None
    assert calls == [], "patched methods must not reach the builder"


def test_frontend_noop_is_idempotent(monkeypatch):
    module, FakeSemantic, _ = _fake_semantic()
    monkeypatch.setattr(triton_pdl, "_semantic_module", lambda: module)
    triton_pdl._install_frontend_noop()
    first = FakeSemantic.grid_dependency_wait
    triton_pdl._install_frontend_noop()
    assert FakeSemantic.grid_dependency_wait is first


def test_frontend_noop_tolerates_absent_semantic(monkeypatch):
    monkeypatch.setattr(
        triton_pdl, "_semantic_module", lambda: types.SimpleNamespace()
    )
    assert triton_pdl._install_frontend_noop() == ()


def test_frontend_noop_tolerates_unimportable_triton(monkeypatch):
    def boom():
        raise ImportError("no triton")

    monkeypatch.setattr(triton_pdl, "_semantic_module", boom)
    assert triton_pdl._install_frontend_noop() == ()


def test_asm_stripper_drops_both_instructions(monkeypatch):
    seen = {}

    def make_amdgcn(src, metadata, options):
        seen["src"] = src
        return "asm"

    class Backend:
        pass

    Backend.make_amdgcn = staticmethod(make_amdgcn)
    monkeypatch.setattr(
        triton_pdl, "_hcu_compiler_module",
        lambda: types.SimpleNamespace(HIPBackend=Backend),
    )

    assert triton_pdl._install_asm_stripper() == "make_amdgcn"
    body = (
        "  s_endpgm\n"
        "  griddepcontrol.launch_dependents\n"
        "  griddepcontrol.wait\n"
        "  s_nop 0\n"
    )
    Backend.make_amdgcn(body, None, None)
    assert "griddepcontrol." not in seen["src"]
    assert "s_endpgm" in seen["src"] and "s_nop 0" in seen["src"]


def test_asm_stripper_tolerates_missing_backend(monkeypatch):
    monkeypatch.setattr(
        triton_pdl, "_hcu_compiler_module", lambda: types.SimpleNamespace()
    )
    assert triton_pdl._install_asm_stripper() is None


def test_install_raises_when_neither_layer_available(monkeypatch):
    monkeypatch.setattr(
        triton_pdl, "_semantic_module", lambda: types.SimpleNamespace()
    )
    monkeypatch.setattr(
        triton_pdl, "_hcu_compiler_module", lambda: types.SimpleNamespace()
    )
    monkeypatch.delattr(triton_pdl.install, "_hcu_pdl_done", raising=False)
    with pytest.raises(RuntimeError, match="triton layout changed"):
        triton_pdl.install()
