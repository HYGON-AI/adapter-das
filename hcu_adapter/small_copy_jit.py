# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging


def adapt_kernel(kernel, jit_type):
    if getattr(kernel, "_hcu_small_copy_standard_jit", None) is not None:
        return False
    if (type(kernel).__module__, type(kernel).__name__) != (
        "triton.backends.hcu.jit", "FastJITFunction"
    ):
        return False
    standard = jit_type(kernel.fn, **kernel._init_kwargs)
    kernel.run = standard.run
    kernel._hcu_small_copy_standard_jit = standard
    return True


def install():
    import torch

    if torch.version.hip is None:
        return
    from sglang.kernels.ops.memory import small_copy
    from triton.runtime.jit import JITFunction
    from hcu_adapter import registry

    kernel = small_copy._small_copy_kernel
    if not adapt_kernel(kernel, JITFunction):
        return
    registry.register(
        "small_copy", "standard-jit",
        target="sglang.kernels.ops.memory.small_copy._small_copy_kernel.run",
        mechanism=registry.MECH_REBIND,
        detail="retain upstream kernel and aliases; use standard Triton tuple argument specialization",
    )
    logging.getLogger("hcu.adapter").warning(
        "HCU adapter: small_copy uses standard Triton JIT for dynamic tuple arguments"
    )
