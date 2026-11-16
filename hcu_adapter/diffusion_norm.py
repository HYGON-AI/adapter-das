# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import logging

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
    logging.getLogger("hcu.adapter").info(
        "HCU adapter: diffusion norm-scale-shift CUDA dispatch registered"
    )
