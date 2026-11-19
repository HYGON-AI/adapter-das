# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

_MISSING = object()
_MARKER = "_hcu_small_copy_standard_jit"


def adapt_kernel(kernel, jit_type, *, register=None):
    if getattr(kernel, _MARKER, None) is not None:
        return False
    if (type(kernel).__module__, type(kernel).__name__) != (
        "triton.backends.hcu.jit", "FastJITFunction"
    ):
        return False
    namespace = vars(kernel)
    if type(namespace) is not dict:
        raise TypeError("small_copy requires a writable kernel namespace")
    previous = {name: namespace.get(name, _MISSING) for name in ("run", _MARKER)}
    standard = jit_type(kernel.fn, **kernel._init_kwargs)
    replacements = {"run": standard.run, _MARKER: standard}

    def check_owner():
        if any(namespace.get(name, _MISSING) is not value
               for name, value in previous.items()):
            raise RuntimeError("small_copy kernel changed before publication")

    check_owner()
    if register is not None:
        register()
    check_owner()
    try:
        # Plain namespace writes preserve aliases and avoid host setters.
        namespace.update(replacements)
    except BaseException:
        for name, value in replacements.items():
            if namespace.get(name, _MISSING) is value:
                if previous[name] is _MISSING:
                    namespace.pop(name, None)
                else:
                    namespace[name] = previous[name]
        raise
    return True


def install():
    import torch

    if torch.version.hip is None:
        return
    from sglang.kernels.ops.memory import small_copy
    from triton.runtime.jit import JITFunction
    from hcu_adapter import registry

    kernel = small_copy._small_copy_kernel

    def register_binding():
        registry.register(
            "small_copy", "standard-jit",
            target="sglang.kernels.ops.memory.small_copy._small_copy_kernel.run",
            mechanism=registry.MECH_REBIND,
            detail="retain upstream kernel and aliases; use standard Triton tuple argument specialization",
        )

    with registry.registration_transaction():
        if not adapt_kernel(kernel, JITFunction, register=register_binding):
            return
    logging.getLogger("hcu.adapter").warning(
        "HCU adapter: small_copy uses standard Triton JIT for dynamic tuple arguments"
    )
