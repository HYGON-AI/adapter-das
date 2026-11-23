# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Install HCU implementations behind SGLang's diffusion RoPE custom ops."""

import logging
from importlib import import_module

from . import registry


_LIBRARY = None


def install():
    global _LIBRARY
    import torch

    if torch.version.hip is None or _LIBRARY is not None:
        return

    from sgl_kernel import diffusion_rope_cpp as hip

    modules = {
        "qknorm_complex_rope_jit": (
            ("qknorm_complex_rope_cuda", hip.qknorm_complex_rope),
            ("qknorm_complex_rope_pack_", hip.qknorm_complex_rope_pack_),
        ),
        "qknorm_complex_rope_triton": (("qknorm_complex_rope", hip.qknorm_complex_rope),),
        "qknorm_complex_rope_kv_triton": (("qknorm_complex_rope_kv", hip.qknorm_complex_rope_kv),),
        "complex_rope_triton": (("fused_complex_rope", hip.fused_complex_rope),),
    }
    bindings = []
    for module, ops in modules.items():
        qualified = f"sglang.kernels.ops.diffusion.rope.{module}"
        try:
            import_module(qualified)
        except ModuleNotFoundError as exc:
            if exc.name and (exc.name == qualified or qualified.startswith(exc.name + ".")):
                continue
            raise
        bindings.extend(ops)
    if not bindings:
        return

    library = torch.library.Library("sglang", "IMPL", "CUDA")
    for name, impl in bindings:
        library.impl(name, impl, allow_override=True)
        registry.register(
            f"diffusion_{name}", impl,
            target=f"sglang::{name}/CUDA", mechanism=registry.MECH_SEAM,
            detail="HCU HIP kernel or Torch fallback behind the upstream schema",
        )
    _LIBRARY = library
    logging.getLogger("hcu.adapter").info(
        "HCU adapter: diffusion RoPE CUDA dispatch registered (%d ops)", len(bindings)
    )
