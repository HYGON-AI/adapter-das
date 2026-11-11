# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Probe for the optional native (hipcc) extension, BACKEND_CATALOG "cpp" tier.

The shadow package ships as a pure-python wheel by default; building
setup_hip.py in a DTK container adds ``sgl_kernel/native_ops*.so``, whose
TORCH_LIBRARY(sgl_kernel) registrations mirror the official wheel exactly.
Import success IS the probe: the .so only exists when it was built, and a
successful import means every schema in csrc/common_extension_hcu.cc is
registered.

Tier order in every consumer: native (if present) -> triton -> torch emu.
Callers must keep their own precondition guards (contiguity, dtype, layout
constraints) BEFORE dispatching to native — the native TORCH_CHECKs mirror
upstream and raise RuntimeError on violation, while the triton/emu tiers
accept a slightly wider face; a native-only crash on an input the fallback
tiers handle would be a tier divergence.
"""

from __future__ import annotations

import torch

_STATE = None  # None = unprobed, then True/False

_OPS = torch.ops.sgl_kernel


def native_available() -> bool:
    """True iff sgl_kernel.native_ops imports (the cpp tier is installed)."""
    global _STATE
    if _STATE is None:
        try:
            import sgl_kernel.native_ops  # noqa: F401  (import = registration)
        except Exception:
            _STATE = False
        else:
            _STATE = True
    return _STATE


def native_op(name: str):
    """torch.ops.sgl_kernel.<name> when the cpp tier is present, else None."""
    return getattr(_OPS, name, None) if native_available() else None
