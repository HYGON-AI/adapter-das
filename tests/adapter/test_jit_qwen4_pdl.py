# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""jit_qwen4 / triton_pdl contracts (CPU only, no GPU).

Covers the two v0.5.22 Qwen3.8-Flash-Next boot failures:

  - the elementwise JIT family structured-binds into `fp32x2_t`, a HIP vector
    type with an anonymous union -> hipcc "cannot decompose class type ...".
    jit_qwen4 rewrites every site to `.x/.y` member reads; grouped_gemma
    aborts scheduler init before weights load, hc_combine aborts decode graph
    capture.
  - `create_grid_dependency_*` has no HCU lowering -> `PassManager::run
    failed` inside `ConvertTritonHCUToLLVM`. triton_pdl neutralises the
    frontend semantic methods (and keeps the asm stripper for wheels that
    lower the ops).
"""

import re
import types

import pytest

from hcu_adapter import jit_qwen4, registry, triton_pdl

# The files jit_qwen4 covers, and how many bindings each one really carries.
# hc_combine repeats its two needle groups in a second kernel body.
_COVERED = {
    "elementwise/grouped_gemma_rmsnorm.cuh": 3,
    "elementwise/hc_combine.cuh": 8,
    "elementwise/hc_combine_decode.cuh": 5,
    "elementwise/rmsnorm.cuh": 9,
}

_BINDING = re.compile(r"const auto \[[^\]]*\] = cast<fp32x2_t>")


def _recorder():
    records = []

    def register(*args, **kwargs):
        records.append((args, kwargs))

    return records, register


def _tree(tmp_path, files):
    root = tmp_path / "bundle"
    for relpath, body in files.items():
        target = root / "csrc" / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return root


def test_every_covered_file_is_declared():
    assert set(jit_qwen4._PATCHES) == set(_COVERED)


def test_needles_are_allowlisted():
    """Every needle must be governance-reviewed in TEXT_PATCH_ALLOWLIST."""
    missing = []
    for pairs in jit_qwen4._PATCHES.values():
        for needle, _ in pairs:
            key = ("rms_norm", needle.lstrip("\n").splitlines()[0])
            if key not in registry.TEXT_PATCH_ALLOWLIST:
                missing.append(key)
    assert not missing, missing


def test_anchored_needles_do_not_shadow_each_other():
    """A shallow-indented needle must not eat a deeper one's tail.

    `hc_combine_decode.cuh` carries the same statement at 4 and 6 spaces. An
    unanchored 4-space needle matches the 6-space line too (consuming its extra
    indent), so the deeper site gets rewritten by the wrong replacement and its
    own needle then goes missing.
    """
    pairs = jit_qwen4._PATCHES["elementwise/hc_combine_decode.cuh"]
    text = (
        "    const auto [x, y] = cast<fp32x2_t>(combined[i]);\n"
        "      const auto [x, y] = cast<fp32x2_t>(combined[i]);\n"
        "      const auto [wx, wy] = cast<fp32x2_t>(weight_values[i]);\n"
    )
    compiled = jit_qwen4._compile(pairs)
    # order: residual pair, 4-space combined, 6-space combined+weight
    shallow, deep = compiled[1], compiled[2]

    once = shallow[0].findall(text)
    assert len(once) == 1, f"shallow needle matched {len(once)} sites"
    patched = shallow[0].sub(lambda _m: shallow[1], text)
    assert deep[0].search(patched) is not None, "deep needle lost its target"
    assert all(p.pattern.startswith("^") for p, _ in compiled)
    assert all(p.flags & re.MULTILINE for p, _ in compiled)


def test_anchoring_matches_at_file_start():
    """`^` anchors at offset 0 too, which a leading newline needle would miss."""
    pair = jit_qwen4._PATCHES["elementwise/grouped_gemma_rmsnorm.cuh"][0]
    compiled = jit_qwen4._compile([pair])
    text = pair[0] + "\n"
    assert compiled[0][0].search(text) is not None


def test_patch_rewrites_every_site(tmp_path):
    """Synthetic tree carrying exactly the real binding count per file."""
    files = {}
    for relpath, expected in _COVERED.items():
        pairs = jit_qwen4._PATCHES[relpath]
        # each needle group is a contiguous block; repeat the whole set until
        # the synthetic file reaches the real binding count
        per_round = sum(
            len(_BINDING.findall(needle)) for needle, _ in pairs
        )
        rounds = max(1, expected // per_round)
        blocks = []
        for _ in range(rounds):
            for needle, _ in pairs:
                blocks.extend(needle.lstrip("\n").splitlines())
        files[relpath] = "\n".join(blocks) + "\n"
        assert len(_BINDING.findall(files[relpath])) == expected, relpath

    root = _tree(tmp_path, files)
    records, register = _recorder()
    jit_qwen4._apply(root, register)

    for relpath in _COVERED:
        body = (root / "csrc" / relpath).read_text()
        assert not _BINDING.search(body), f"{relpath} still has a structured binding"
    assert records, "expected allowlist records"
    for args, kwargs in records:
        assert args[0] == "rms_norm"
        assert kwargs["mechanism"] == registry.MECH_TEXT
        assert kwargs["target"].startswith("text:sglang/kernels/jit/")


def test_patch_is_idempotent(tmp_path):
    src = "/public/home/yaoht/sglang/github/sglang/python/sglang/kernels/jit"
    import pathlib

    if not pathlib.Path(src).is_dir():
        pytest.skip("community sglang tree not mounted")
    files = {
        relpath: (pathlib.Path(src) / "csrc" / relpath).read_text()
        for relpath in _COVERED
    }
    root = _tree(tmp_path, files)
    records, register = _recorder()
    jit_qwen4._apply(root, register)
    once = {r: (root / "csrc" / r).read_text() for r in _COVERED}
    calls = len(records)

    jit_qwen4._apply(root, register)  # second install must be a no-op
    assert {r: (root / "csrc" / r).read_text() for r in _COVERED} == once
    assert len(records) == calls


def test_absent_files_are_skipped(tmp_path):
    root = tmp_path / "bundle"
    (root / "csrc").mkdir(parents=True)
    records, register = _recorder()
    jit_qwen4._apply(root, register)  # must not raise
    assert records == []


def test_missing_needle_raises(tmp_path):
    root = _tree(
        tmp_path,
        {relpath: "// upstream dropped the kernel\n" for relpath in _COVERED},
    )
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
