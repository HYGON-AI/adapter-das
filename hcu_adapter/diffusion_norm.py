# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module

from . import registry


_LIBRARY = None


def install():
    global _LIBRARY
    import torch

    if torch.version.hip is None or _LIBRARY is not None:
        return
    from sgl_kernel.norm_scale_shift_triton import fused_norm_scale_shift

    from sgl_kernel.residual_norm_scale_shift_triton import fused_scale_residual_norm_scale_shift

    library = torch.library.Library("sglang", "IMPL", "CUDA")
    library.impl("fused_norm_scale_shift", fused_norm_scale_shift)
    library.impl("fused_scale_residual_norm_scale_shift", fused_scale_residual_norm_scale_shift)
    _LIBRARY = library
    registry.register(
        "diffusion_norm_scale_shift", fused_norm_scale_shift,
        target="sglang::fused_norm_scale_shift/CUDA", mechanism=registry.MECH_SEAM,
        detail="HCU Triton LayerNorm modulation with Torch fallback; upstream schema and aliases preserved",
    )
    registry.register(
        "diffusion_residual_norm_scale_shift", fused_scale_residual_norm_scale_shift,
        target="sglang::fused_scale_residual_norm_scale_shift/CUDA", mechanism=registry.MECH_SEAM,
        detail="HCU Triton residual + layer/RMS norm modulation; upstream tuple and aliases preserved",
    )
    _install_layernorm_modulate(library)
    logging.getLogger("hcu.adapter").info(
        "HCU adapter: diffusion norm-scale-shift CUDA dispatch registered"
    )


def _install_layernorm_modulate(library):
    from sgl_kernel.norm_scale_shift_triton import fused_layernorm_modulate

    upstream = "sglang.kernels.kda_kernels.layernorm_modulate_triton"
    try:
        import_module(upstream)
    except ModuleNotFoundError as exc:
        if exc.name and (exc.name == upstream or upstream.startswith(exc.name + ".")):
            return
        raise

    op_name = "triton_fused_layernorm_modulate"
    library.impl(op_name, fused_layernorm_modulate, allow_override=True)
    registry.register(
        "diffusion_layernorm_modulate", fused_layernorm_modulate,
        target=f"sglang::{op_name}/CUDA", mechanism=registry.MECH_SEAM,
        detail="BF16 modulation on HCU with the upstream schema and exactness gates",
    )
