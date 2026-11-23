# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in bridges used by GLM-5.3-Flash blockwise FP8 on HIP."""

from hcu_adapter import hcu_env


def install(_he=None):
    env = _he or hcu_env

    if env.enabled("HCU_ORDERED_TP8_BF16_REDUCE"):
        from hcu_adapter.bridges import sglang_allreduce

        sglang_allreduce.install_ordered_tp_reduce()

    if env.enabled("HCU_GLM53_FIXED_KDA_PROJECTION"):
        from hcu_adapter.features.glm53_determinism import kda_invariant_projection

        kda_invariant_projection.install()

    if env.enabled("HCU_DSA_ORDERED_SHORT_TOPK"):
        from hcu_adapter import dsa_topk

        dsa_topk.install_ordered_short_kpool()

    if env.enabled("HCU_GLM53_FIXED_MHC_STATS"):
        from hcu_adapter.features.glm53_determinism import mhc_invariant_stats

        mhc_invariant_stats.install()

    if env.enabled("HCU_GLM53_FIXED_EAGLE_PROJECTIONS"):
        from hcu_adapter.features.glm53_determinism import glm53_invariant_eagle

        glm53_invariant_eagle.install()

    if env.enabled("HCU_GLM53_FIXED_DSA_MATMUL"):
        from hcu_adapter.features.glm53_determinism import glm53_invariant_mla

        glm53_invariant_mla.install()
