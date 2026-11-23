# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free checks of the GLM-5.3 deterministic opt-ins."""

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hcu_adapter import hcu_config, hcu_env
from hcu_adapter.bridges import sglang_glm53_determinism


_CATALOG = {
    "kda_projection": ("fixed_tile", "HCU_GLM53_FIXED_KDA_PROJECTION"),
    "dsa_short_topk": ("ordered", "HCU_DSA_ORDERED_SHORT_TOPK"),
    "glm_mhc_stats": ("fixed_stats", "HCU_GLM53_FIXED_MHC_STATS"),
    "glm_eagle_projections": ("fixed", "HCU_GLM53_FIXED_EAGLE_PROJECTIONS"),
    "glm_mla_matmul": ("fixed", "HCU_GLM53_FIXED_DSA_MATMUL"),
}
_SWITCHES = tuple(var for _, var in _CATALOG.values()) + (
    "HCU_ORDERED_TP8_BF16_REDUCE",
    "HCU_KDA_FIXED_SAFE_GATE_ANCHOR",
    "HCU_FIXED_BLOCK_FP8_DENSE",
)
_FEATURES = "hcu_adapter.features.glm53_determinism"


class DeterminismOptInTests(unittest.TestCase):
    def setUp(self):
        self.contexts = ExitStack()
        self.addCleanup(self.contexts.close)
        self.contexts.enter_context(patch.dict(os.environ, {"HCU_ADAPTER_HOST": "none"}, clear=True))
        self.contexts.enter_context(patch.object(hcu_config, "_STATE", {"loaded": False}))
        directory = self.contexts.enter_context(TemporaryDirectory())
        self.config = Path(directory) / "adapter.json"
        self.contexts.enter_context(patch.object(hcu_config, "_candidate_paths", return_value=[str(self.config)]))

    def test_opt_in_and_explicit_disable(self):
        for switch in _SWITCHES:
            with self.subTest(switch=switch), patch.dict(os.environ):
                self.config.write_text(json.dumps({"version": 1}))
                hcu_config.load(force=True)
                self.assertEqual(hcu_env.REGISTRY[switch]["env"], switch)
                self.assertEqual(hcu_env.resolve(switch), ("0", "default"))
                self.assertFalse(hcu_env.enabled(switch))

                for op, (backend, key) in _CATALOG.items():
                    if key == switch:
                        entry = hcu_config.BACKEND_CATALOG[op]
                        self.assertEqual(entry["backends"][backend]["status"], "candidate")
                        self.assertEqual(entry["vars"][backend], {switch: "1"})
                        self.config.write_text(json.dumps({"version": 1, "ops": {op: backend}}))
                        hcu_config.load(force=True)
                        self.assertEqual(hcu_env.resolve(switch), ("1", "config"))
                        self.assertTrue(hcu_env.enabled(switch))

                os.environ[switch] = "1"
                self.assertEqual(hcu_env.resolve(switch), ("1", "explicit"))
                self.assertTrue(hcu_env.enabled(switch))
                os.environ[switch] = "0"
                self.assertEqual(hcu_env.resolve(switch), ("0", "explicit"))
                self.assertFalse(hcu_env.enabled(switch))

    def test_deployment_recipe_selects_all_fixed_paths(self):
        self.config.write_text(json.dumps({
            "version": 1,
            "ops": {"glm53_block_fp8_determinism": "fixed", "fused_moe": "triton"},
        }))
        hcu_config.load(force=True)
        for switch in _SWITCHES:
            with self.subTest(switch=switch), patch.dict(os.environ, {switch: "0"}):
                self.assertEqual(hcu_config.lookup(switch), "1")
                self.assertEqual(hcu_env.resolve(switch), ("0", "explicit"))
            self.assertEqual(hcu_env.resolve(switch), ("1", "config"))
        self.assertEqual(hcu_env.resolve("HCU_FUSED_MOE_IMPL"), ("triton", "config"))

    def test_bridge_calls_only_selected_installers(self):
        routes = {
            "HCU_ORDERED_TP8_BF16_REDUCE": (
                "hcu_adapter.bridges", "sglang_allreduce", "install_ordered_tp_reduce",
            ),
            "HCU_DSA_ORDERED_SHORT_TOPK": ("hcu_adapter", "dsa_topk", "install_ordered_short_kpool"),
            "HCU_GLM53_FIXED_KDA_PROJECTION": (_FEATURES, "kda_invariant_projection", "install"),
            "HCU_GLM53_FIXED_MHC_STATS": (_FEATURES, "mhc_invariant_stats", "install"),
            "HCU_GLM53_FIXED_EAGLE_PROJECTIONS": (_FEATURES, "glm53_invariant_eagle", "install"),
            "HCU_GLM53_FIXED_DSA_MATMUL": (_FEATURES, "glm53_invariant_mla", "install"),
        }
        calls = []
        for flag, (parent, name, entry) in routes.items():
            installer = SimpleNamespace(**{entry: lambda key=flag: calls.append(key)})
            self.contexts.enter_context(patch.object(importlib.import_module(parent), name, installer, create=True))

        sglang_glm53_determinism.install()
        self.assertEqual(calls, [])
        os.environ.update(dict.fromkeys(_SWITCHES, "0"))
        sglang_glm53_determinism.install()
        self.assertEqual(calls, [])
        for flag in routes:
            with self.subTest(switch=flag), patch.dict(os.environ, {flag: "1"}):
                sglang_glm53_determinism.install()
                self.assertEqual(calls, [flag])
                calls.clear()
