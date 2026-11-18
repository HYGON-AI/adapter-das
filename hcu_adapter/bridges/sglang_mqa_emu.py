# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""gfx936 MQA-logits fp8 emu (paged/ragged torch fallback auto-rule)

Extracted verbatim from bridges/sglang.py install() under layout
governance 2026-11-18; zero logic change. Section notes below as #
comments (glue-numerics strips comments only).
"""
from __future__ import annotations

import logging
import os


def install(_he=None) -> None:
    # gfx936-class HCUs have no fp8 matrix unit and the DTK aiter MQA-logits
    # kernels hard-gate on gfx942/950: emulate paged/ragged fp8 MQA logits in
    # torch (dequant -> bf16 GEMM, fp32 scales). Legacy page_size=1 only.
    # hcu_env auto-rule: emulation ON only where the arch lacks fp8;
    # explicit HCU_MQA_EMU=0/1 wins (debug).
    _mqa_on = _he.enabled("HCU_MQA_EMU") if _he else \
        os.environ.get("HCU_MQA_EMU") == "1"
    if _mqa_on:
        try:
            import hcu_adapter.features.dsa.mqa_logits_emu as _mqa_emu

            _mqa_emu.install()
        except ModuleNotFoundError:
            pass  # no sglang importable (adapter-only tooling) — emu irrelevant
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: MQA-logits emu not installed", exc_info=True
            )
