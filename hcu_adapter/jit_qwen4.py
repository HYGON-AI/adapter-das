# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT grouped Gemma RMSNorm portability patch — DTK hipcc vs CUDA-isms.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.

Upstream v0.5.22 JIT-compiles `Qwen4ExpPLEGroupedNorm` (Qwen3.8-Flash-Next PLE
side state) from `sglang/kernels/jit/csrc/elementwise/grouped_gemma_rmsnorm.cuh`
on first call. gfx938's DTK clang-based hipcc rejects the CUDA-ism that the rest
of this family already had to drop elsewhere (jit_rope / jit_warp / jit_dpskmeta
all carry the same needle class):

    const auto [x, y] = cast<fp32x2_t>(input_vec[j][i]);
    error: cannot decompose class type 'HIP_vector_type<float, 2> const'
           because it has an anonymous union member

`fp32x2_t` is a HIP vector type with an anonymous union, so a structured
binding into it is ill-formed here. Replacing the binding with explicit `.x/.y`
member reads is behaviour-identical on both platforms (same lanes, same f32
arithmetic, no reassociation).

Without the patch the whole PLE module family dies at build time, which aborts
scheduler init before weights finish loading:

    RuntimeError: Failed to build JIT module
      sgl_kernel_jit_grouped_gemma_rmsnorm_2560_false_bf16_t

Same text-patch mechanics as jit_rope: every needle applies idempotently
(already-present replacement -> skip that needle, continue); a missing needle in
an existing file raises loud (upstream changed; re-derive), an absent file skips
quietly (the tree predates the module).
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "elementwise/grouped_gemma_rmsnorm.cuh"

# (needle, replacement); all three are the same ABI fact at different sites.
_PATCHES = (
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
)


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""
    import hcu_adapter.registry as _reg
    from hcu_adapter.atomic_io import atomic_write

    p = root / "csrc" / _RELPATH
    if not p.exists():
        LOG.info("jit_qwen4: %s absent (tree predates the module)", _RELPATH)
        return

    text = p.read_text(encoding="utf-8")
    applied = 0
    for needle, repl in _PATCHES:
        if repl in text:
            applied += 1  # already patched (spawn'd child / second install)
            continue
        if needle not in text:
            raise RuntimeError(
                f"jit_qwen4: needle missing in {_RELPATH}: {needle[:60]!r} -- "
                "upstream changed; re-derive (governance §3)"
            )
        text = text.replace(needle, repl)
        applied += 1
        register(
            "rms_norm",
            "gfx93x-hipcc-portability",
            target=f"text:sglang/kernels/jit/{_RELPATH}",
            mechanism=_reg.MECH_TEXT,
            detail=needle,
        )

    if applied:
        atomic_write(p, text)
    LOG.warning(
        "HCU adapter: JIT grouped_gemma_rmsnorm portability patch installed "
        "(%d sites)", applied,
    )


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
