# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT elementwise HIP portability patch — DTK hipcc vs CUDA-isms.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.

Upstream v0.5.22 JIT-compiles several elementwise kernels from
`sglang/kernels/jit/csrc/elementwise/`. gfx938's DTK clang-based hipcc rejects
the CUDA-ism this family has had to drop everywhere else (jit_rope / jit_warp /
jit_dpskmeta all carry the same needle class):

    const auto [x, y] = cast<fp32x2_t>(some_vec[i]);
    error: cannot decompose class type 'HIP_vector_type<float, 2> const'
           because it has an anonymous union member

`fp32x2_t` is a HIP vector type with an anonymous union, so a structured
binding into it is ill-formed here. Replacing the binding with explicit `.x/.y`
member reads is behaviour-identical on both platforms (same lanes, same f32
arithmetic, no reassociation).

Four files are covered, chosen by which ones actually reach the compiler on
this stack (each was hit in a live boot, not guessed):

  elementwise/grouped_gemma_rmsnorm.cuh  Qwen4ExpPLEGroupedNorm (Qwen3.8 PLE)
  elementwise/hc_combine.cuh             mHC pre-combine (decode + prefill arms)
  elementwise/hc_combine_decode.cuh      mHC decode combine
  elementwise/rmsnorm.cuh                plain RMSNorm CTA kernel

Without the patch the module dies at build time, which aborts whatever phase
first needs it — grouped_gemma before weights finish loading, hc_combine during
decode CUDA-graph capture:

    RuntimeError: Failed to build JIT module
      sgl_kernel_jit_grouped_gemma_rmsnorm_2560_false_bf16_t
    RuntimeError: Failed to build JIT module
      sgl_kernel_jit_hc_combine_4_2560_false_bf16_t

Same text-patch mechanics as jit_rope: every needle applies idempotently
(already-present replacement -> skip that needle, continue); a missing needle in
an existing file raises loud (upstream changed; re-derive), an absent file skips
quietly (the tree predates the module).

Note on rmsnorm.cuh: some deployments never JIT-compile it (the triton tier
serves `HCU_RMS_NORM_IMPL=triton`), so its needles can sit unused — the patch is
still required for the AOT/CTA arm and keeps the whole family consistent.
"""

import logging
import re

LOG = logging.getLogger("hcu.adapter")

# stamped into every replacement; presence in the SOURCE text as read means an
# earlier run already patched this file (see _apply_file)
_HCU_MARKER = "HCU: float2 not decomposable"

# file -> ((needle, replacement), ...)
_PATCHES = {
    "elementwise/grouped_gemma_rmsnorm.cuh": (
        (
            "      const auto [x, y] = cast<fp32x2_t>(input_vec[j][i]);",
            "      const fp32x2_t _xy = cast<fp32x2_t>(input_vec[j][i]);"
            "  // HCU: float2 not decomposable\n"
            "      const auto x = _xy.x, y = _xy.y;",
        ),
        (
            "      const auto [ix, iy] = cast<fp32x2_t>(input_vec[j][i]);",
            "      const fp32x2_t _ixy = cast<fp32x2_t>(input_vec[j][i]);"
            "  // HCU: float2 not decomposable\n"
            "      const auto ix = _ixy.x, iy = _ixy.y;",
        ),
        (
            "      const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j][i]);",
            "      const fp32x2_t _wxy = cast<fp32x2_t>(weight_vec[j][i]);"
            "  // HCU: float2 not decomposable\n"
            "      const auto wx = _wxy.x, wy = _wxy.y;",
        ),
    ),
    # Two kernel bodies in this file (prefill + decode arms) repeat the same
    # four bindings, so each needle matches twice at different indentation.
    "elementwise/hc_combine.cuh": (
        (
            "        const auto [nx, ny] = cast<fp32x2_t>(n_vec[j][i]);\n"
            "        const auto [wx, wy] = cast<fp32x2_t>(w_vec[i]);\n"
            "        sum += nx * wx + ny * wy;",
            "        const auto _nxy = cast<fp32x2_t>(n_vec[j][i]);"
            "  // HCU: float2 not decomposable\n"
            "        const auto _wxy = cast<fp32x2_t>(w_vec[i]);\n"
            "        sum += _nxy.x * _wxy.x + _nxy.y * _wxy.y;",
        ),
        (
            "      const auto [rx, ry] = cast<fp32x2_t>(r_vec[i]);\n"
            "      const auto [yx, yy] = cast<fp32x2_t>(y_vec[i]);\n"
            "      out_vec[i] = cast<Float2>(fp32x2_t{rx + a * yx, ry + a * yy});",
            "      const auto _rxy = cast<fp32x2_t>(r_vec[i]);"
            "  // HCU: float2 not decomposable\n"
            "      const auto _yxy = cast<fp32x2_t>(y_vec[i]);\n"
            "      out_vec[i] = cast<Float2>(\n"
            "          fp32x2_t{_rxy.x + a * _yxy.x, _rxy.y + a * _yxy.y});",
        ),
    ),
    "elementwise/hc_combine_decode.cuh": (
        (
            "    const auto [rx, ry] = cast<fp32x2_t>(residual_values[i]);\n"
            "    const auto [yx, yy] = cast<fp32x2_t>(block_values[i]);",
            "    const auto _rxy = cast<fp32x2_t>(residual_values[i]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto _yxy = cast<fp32x2_t>(block_values[i]);\n"
            "    const auto rx = _rxy.x, ry = _rxy.y;\n"
            "    const auto yx = _yxy.x, yy = _yxy.y;",
        ),
        (
            "    const auto [x, y] = cast<fp32x2_t>(combined[i]);",
            "    const auto _xy = cast<fp32x2_t>(combined[i]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto x = _xy.x, y = _xy.y;",
        ),
        (
            "      const auto [x, y] = cast<fp32x2_t>(combined[i]);\n"
            "      const auto [wx, wy] = cast<fp32x2_t>(weight_values[i]);",
            "      const auto _xy = cast<fp32x2_t>(combined[i]);"
            "  // HCU: float2 not decomposable\n"
            "      const auto _wxy = cast<fp32x2_t>(weight_values[i]);\n"
            "      const auto x = _xy.x, y = _xy.y;\n"
            "      const auto wx = _wxy.x, wy = _wxy.y;",
        ),
    ),
    "elementwise/rmsnorm.cuh": (
        (
            "    const auto [x, y] = cast<fp32x2_t>(input_first[j]);",
            "    const auto _xy = cast<fp32x2_t>(input_first[j]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto x = _xy.x, y = _xy.y;",
        ),
        (
            "    const auto [x, y] = cast<fp32x2_t>(input_second[j]);",
            "    const auto _xy2 = cast<fp32x2_t>(input_second[j]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto x = _xy2.x, y = _xy2.y;",
        ),
        (
            "    const auto [ix, iy] = cast<fp32x2_t>(input_first[j]);\n"
            "    const auto [wx, wy] = cast<fp32x2_t>(weight_first[j]);",
            "    const auto _ixy = cast<fp32x2_t>(input_first[j]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto _wxy = cast<fp32x2_t>(weight_first[j]);\n"
            "    const auto ix = _ixy.x, iy = _ixy.y;\n"
            "    const auto wx = _wxy.x, wy = _wxy.y;",
        ),
        (
            "    const auto [ix, iy] = cast<fp32x2_t>(input_second[j]);\n"
            "    const auto [wx, wy] = cast<fp32x2_t>(weight_second[j]);",
            "    const auto _ixy2 = cast<fp32x2_t>(input_second[j]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto _wxy2 = cast<fp32x2_t>(weight_second[j]);\n"
            "    const auto ix = _ixy2.x, iy = _ixy2.y;\n"
            "    const auto wx = _wxy2.x, wy = _wxy2.y;",
        ),
        (
            "    const auto [x, y] = cast<fp32x2_t>(input_vec[j]);",
            "    const auto _xyv = cast<fp32x2_t>(input_vec[j]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto x = _xyv.x, y = _xyv.y;",
        ),
        (
            "    const auto [ix, iy] = cast<fp32x2_t>(input_vec[j]);\n"
            "    const auto [wx, wy] = cast<fp32x2_t>(weight_vec[j]);",
            "    const auto _ixyv = cast<fp32x2_t>(input_vec[j]);"
            "  // HCU: float2 not decomposable\n"
            "    const auto _wxyv = cast<fp32x2_t>(weight_vec[j]);\n"
            "    const auto ix = _ixyv.x, iy = _ixyv.y;\n"
            "    const auto wx = _wxyv.x, wy = _wxyv.y;",
        ),
    ),
}


def _compile(pairs):
    """Compile needles into line-anchored (pattern, replacement) pairs.

    Anchoring each needle at a line start is required: without it a statement
    indented 4 spaces also matches the *tail* of the identical statement
    indented 6 spaces (the shorter indent consumes the extra two characters),
    so patching the shallow copy silently rewrites the deep one and the next
    needle can no longer find its target. `re.MULTILINE` + `^` anchors at file
    start too, which a leading "\\n" would miss.

    The pattern is for MATCHING only. Registration always uses the original
    needle text: `re.escape` turns spaces into `\\ `, so a compiled pattern can
    never equal a TEXT_PATCH_ALLOWLIST entry (which holds source text), and
    `jit_sources.prepare` would reject the whole bundle as unreviewed.
    """
    return tuple(
        (
            re.compile(r"^" + re.escape(needle) if not needle.startswith("^")
                       else needle, re.MULTILINE | re.DOTALL),
            repl,
            needle,
        )
        for needle, repl in pairs
    )


def _apply_file(path, relpath, pairs, register):
    """Apply one file's needles; return (sites_touched, skipped_reason)."""
    import hcu_adapter.registry as _reg
    from hcu_adapter.atomic_io import atomic_write

    if not path.exists():
        return 0, "absent"

    original = path.read_text(encoding="utf-8")
    # Idempotency must be judged on the text as READ, never on `text` while it
    # is being rewritten: the replacements carry the HCU marker, so a needle
    # that goes missing later in this same loop would otherwise see a marker
    # this run just wrote and be skipped as "already patched" -- silently
    # publishing a half-patched file instead of failing loud.
    already_patched = _HCU_MARKER in original
    text = original
    applied = 0
    for pattern, repl, needle in _compile(pairs):
        if pattern.search(text) is None:
            if already_patched:
                # second install / spawn'd child: the whole file was patched by
                # an earlier run, so a needle absent now means its replacement
                # is already in place.
                continue
            raise RuntimeError(
                f"jit_elementwise: needle missing in {relpath}: "
                f"{needle.splitlines()[0][:70]!r} -- upstream "
                "changed; re-derive (governance §3)"
            )
        text, n = pattern.subn(lambda _m, _r=repl: _r, text)
        applied += n
        # Registration detail is the ORIGINAL needle's first line (source text,
        # as TEXT_PATCH_ALLOWLIST holds it -- never the escaped pattern). The
        # indentation makes the first line unique per site.
        register(
            "rms_norm",
            "gfx93x-hipcc-portability",
            target=f"text:sglang/kernels/jit/csrc/{relpath}",
            mechanism=_reg.MECH_TEXT,
            detail=needle.splitlines()[0],
        )

    if applied:
        atomic_write(path, text)
    return applied, None


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""
    total = 0
    for relpath, pairs in _PATCHES.items():
        applied, skipped = _apply_file(
            root / "csrc" / relpath, relpath, pairs, register
        )
        if skipped:
            LOG.info(
                "jit_elementwise: %s absent (tree predates the module)", relpath
            )
        elif applied:
            LOG.warning(
                "HCU adapter: JIT %s portability patch installed (%d sites)",
                relpath.rsplit("/", 1)[-1], applied,
            )
        total += applied
    if total:
        LOG.warning(
            "HCU adapter: JIT elementwise HIP portability total %d sites", total
        )


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
