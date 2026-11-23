# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Real bundle-prepare integration for the JIT transforms (CPU only).

The unit tests in test_jit_qwen4_pdl.py compare constant tables to each other;
they cannot catch a transform that *registers* something the allowlist does not
hold, because registration only happens inside a real `prepare()` run against a
real community tree. That gap shipped once: `detail=pattern.pattern`
(re.escape'd) made all 14 elementwise records unapproved, and the failure was
invisible on a warm cache because `prepare()` returned the cached bundle before
validating anything.

These tests need the community SGLang tree and skip without it; pass
HCU_TEST_SGLANG_TREE (same gate as test_jit_sources.py) to point at a checkout.
No default path is baked in: an absolute path from one developer's home would
be dead weight in every other environment and leaks personal environment
details into the repository.
"""

import json
import os
from pathlib import Path

import pytest

from hcu_adapter import jit_qwen4, jit_sources, registry


def _tree():
    """Community JIT source root, from the same gate test_jit_sources uses.

    HCU_TEST_SGLANG_TREE points at the SGLang *repository root* (matching
    test_jit_sources.py), so the JIT tree is rooted under
    python/sglang/kernels/jit. Accepting a path that already ends at the JIT
    root keeps the variable usable for both shapes.
    """
    root = os.environ.get("HCU_TEST_SGLANG_TREE")
    if not root:
        pytest.skip("requires explicitly selected community sglang tree")
    root = Path(root)
    candidates = [root / "python/sglang/kernels/jit", root]
    for candidate in candidates:
        if (candidate / "csrc").is_dir():
            return candidate
    pytest.skip(
        f"HCU_TEST_SGLANG_TREE has no python/sglang/kernels/jit/csrc: {root}"
    )


def test_committed_needles_are_allowlisted():
    """Registration detail must equal the allowlist's source text."""
    missing = []
    for relpath, pairs in jit_qwen4._PATCHES.items():
        for needle, _repl in pairs:
            if ("rms_norm", needle.splitlines()[0]) not in registry.TEXT_PATCH_ALLOWLIST:
                missing.append((relpath, needle.splitlines()[0]))
    assert not missing, missing


def test_cold_prepare_accepts_every_registered_record(tmp_path):
    """Run the real pipeline on the real tree with a cold cache.

    This is the check whose absence let 7d9e560 through: it exercises every
    transform's actual registration against TEXT_PATCH_ALLOWLIST.
    """
    root = _tree()
    cache = tmp_path / "cold"
    bundle, manifest = jit_sources.prepare(root, cache)

    assert bundle.is_dir()
    assert manifest["records"], "expected transforms to register records"
    for args, kwargs in manifest["records"]:
        assert (args[0], kwargs["detail"]) in registry.TEXT_PATCH_ALLOWLIST, (
            f"unapproved record from prepare(): {args[0]!r} "
            f"{str(kwargs['detail'])[:80]!r}"
        )
    # the elementwise family must be among them
    targets = {kwargs["target"] for _args, kwargs in manifest["records"]}
    assert any("grouped_gemma_rmsnorm" in t for t in targets)
    assert any("hc_combine.cuh" in t for t in targets)


def test_warm_bundle_is_revalidated(tmp_path, monkeypatch):
    """A cached bundle must not bypass the allowlist gate.

    Regression: `prepare()` returned the existing bundle before validating, so
    a bundle written by an earlier buggy revision replayed forever with no
    review and no error.
    """
    cache = tmp_path / "cache"
    bundle = cache / "deadbeef"
    (bundle / "csrc").mkdir(parents=True)
    (bundle / "include").mkdir()
    (bundle / "cuda-headers").mkdir()

    poison = {
        "key": "deadbeef",
        "recipe": {"inputs": {}, "transforms": {}},
        "source_root": "/nowhere",
        "outputs": {},
        "forwarders": {},
        "records": [
            [
                ["rms_norm", "gfx93x-hipcc-portability"],
                {"target": "jit-input:x", "detail": "not a reviewed needle"},
            ]
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(poison))

    monkeypatch.setattr(jit_sources, "_verify", lambda *_a, **_k: poison)
    monkeypatch.setattr(jit_sources, "_digest", lambda _data: "deadbeef")

    # prepare() snapshots the community root before consulting the cache, so a
    # minimal but structurally valid tree has to exist for the gate to be the
    # thing that fails.
    root = tmp_path / "jit"
    (root / "csrc").mkdir(parents=True)
    (root / "include").mkdir()
    (root / "csrc" / "probe.cuh").write_text("// probe\n")

    with pytest.raises(RuntimeError, match="unreviewed compiler transformation"):
        jit_sources.prepare(root, cache)
