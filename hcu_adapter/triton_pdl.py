# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Triton PDL semantics on HCU — frontend no-op + GCN asm stripper.

is_cuda semantics let upstream kernels enable Programmatic Dependent Launch
whenever `torch.cuda.get_device_capability()` reports major>=9, which HCU does
(gfx93x -> (9,3)) even though the HIP/ROCm toolchain on this install has no
`griddepcontrol.*` support.

Two distinct failures come from that single fact, and they need two layers:

1. **Frontend (`create_grid_dependency_*` op).** Upstream kernel bodies call
   `tl.extra.cuda.gdc_launch_dependents()` / `gdc_wait()` guarded by
   `ENABLE_PDL`. On the DTK triton wheel the HCU backend's
   `ConvertTritonHCUToLLVM` pass has no lowering for the emitted
   `ttng.grid_dependency_*` ops and the pipeline dies before codegen:

       RuntimeError: PassManager::run failed
         Pipeline failed while executing [`ConvertTritonHCUToLLVM` ...]

   Nothing reaches asm in that case, so an asm-text stripper cannot help.
   Fix: replace the two `TritonSemantic` methods that create the ops with
   no-ops. PDL is a pure scheduling hint -- upstream GPU code waits for the
   prior kernel where it consumes its results when PDL is off, so no-op is
   exactly PDL-off.

2. **Codegen (`griddepcontrol.*` asm).** When the frontend op *is* lowered by
   the wheel's pass pipeline, the gcn->hsaco stage fails with
   "invalid instruction: griddepcontrol.launch_dependents" for every kernel
   compiled with `ENABLE_PDL=[True]` (fused qk rmsnorm-rope-gate, decode
   attention, fla layernorm-gated, ...). Strip both lines from the asm text
   before clang sees them; semantically identical to PDL-off.

Both layers are installed so the same adapter works across triton wheels. Each
is idempotent and independently guarded by its own capability probe: a wheel
whose semantic layer has no `grid_dependency_*` method keeps only the stripper,
and a wheel that already fails earlier only ever hits the frontend no-op.
"""

import logging

LOG = logging.getLogger("hcu.adapter")

# asm lines the GCN stage cannot assemble
_PDL_LINES = (
    "griddepcontrol.launch_dependents",
    "griddepcontrol.wait",
)

# semantic methods that create the ttng grid-dependency ops
_SEMANTIC_METHODS = (
    "grid_dependency_wait",
    "grid_dependency_launch_dependents",
)


def _semantic_module():
    """Indirection so tests can exercise the shim without real triton."""
    from triton.language import semantic

    return semantic


def _hcu_compiler_module():
    """Indirection so tests can exercise the stripper without real triton."""
    import triton.backends.hcu.compiler as hcu_compiler

    return hcu_compiler


def _make_noop(name):
    def _noop(self, *args, **kwargs):
        return None

    _noop.__name__ = name
    _noop.__qualname__ = "TritonSemantic." + name
    _noop._hcu_pdl_noop = True
    return _noop


def _install_frontend_noop():
    """Neutralise the ttng grid-dependency frontend ops; return names done."""
    try:
        semantic = _semantic_module()
    except Exception as exc:  # pragma: no cover - non-triton interpreter
        LOG.warning("HCU adapter: triton semantic not importable (%s)", exc)
        return ()

    owner = getattr(semantic, "TritonSemantic", None)
    if owner is None:
        LOG.warning(
            "HCU adapter: triton semantic has no TritonSemantic; PDL frontend "
            "no-op not installed"
        )
        return ()

    installed = []
    for name in _SEMANTIC_METHODS:
        if not hasattr(owner, name):
            continue
        if not getattr(getattr(owner, name), "_hcu_pdl_noop", False):
            setattr(owner, name, _make_noop(name))
        installed.append(name)

    if not installed:
        LOG.warning(
            "HCU adapter: triton TritonSemantic declares no grid_dependency_* "
            "method; PDL frontend no-op not installed"
        )
        return ()

    LOG.warning(
        "HCU adapter: triton PDL frontend no-op installed (semantic.%s; "
        "HCU cannot lower grid-dependency, semantically = PDL-off)",
        ", ".join(installed),
    )
    return tuple(installed)


def _install_asm_stripper():
    """Drop griddepcontrol.* asm lines at the amdgcn stage; return target."""
    try:
        _hcu = _hcu_compiler_module()
    except Exception as exc:  # pragma: no cover - non-HCU triton
        LOG.warning(
            "HCU adapter: triton hcu compiler not importable (%s); asm "
            "stripper not installed", exc,
        )
        return None

    if getattr(_hcu, "_hcu_pdl_stripped", False):
        return "make_amdgcn"

    owner = getattr(_hcu, "HIPBackend", None) or getattr(
        _hcu, "HCUBackend", None
    )
    if owner is None or not hasattr(owner, "make_amdgcn"):
        LOG.warning(
            "HCU adapter: no HIPBackend/HCUBackend.make_amdgcn in "
            "triton.backends.hcu.compiler -- asm stripper not installed"
        )
        return None
    orig = owner.make_amdgcn

    @staticmethod
    def make_amdgcn_no_pdl(src, metadata, options):
        text = src
        if "griddepcontrol." in text:
            kept = [
                line
                for line in text.splitlines(True)
                if not any(p in line for p in _PDL_LINES)
            ]
            text = "".join(kept)
        return orig(text, metadata, options)

    owner.make_amdgcn = make_amdgcn_no_pdl
    _hcu._hcu_pdl_stripped = True

    LOG.warning(
        "HCU adapter: triton hcu make_amdgcn PDL stripper installed "
        "(gfx936 griddepcontrol unsupported)"
    )
    return "make_amdgcn"


def install():
    from hcu_adapter import registry

    if getattr(install, "_hcu_pdl_done", False):
        return
    semantic_methods = _install_frontend_noop()
    stripped = _install_asm_stripper()
    if not semantic_methods and stripped is None:
        raise RuntimeError(
            "triton_pdl: neither the frontend no-op nor the asm stripper "
            "could be installed -- triton layout changed"
        )

    layers = []
    if semantic_methods:
        layers.append("frontend:" + ",".join(semantic_methods))
    if stripped:
        layers.append("asm:" + stripped)
    targets = []
    if semantic_methods:
        targets.append(
            "triton.language.semantic:TritonSemantic.grid_dependency_*"
        )
    if stripped:
        targets.append("triton.backends.hcu:HCUBackend.make_amdgcn")
    registry.register(
        "triton_pdl",
        "pdl-off",
        target=" + ".join(targets),
        mechanism=registry.MECH_REBIND,
        detail="PDL-off on HCU (no griddepcontrol support): " + "; ".join(layers),
    )
    install._hcu_pdl_done = True
