import importlib.util
import inspect
from pathlib import Path
import unittest
from unittest.mock import Mock

path = Path(__file__).resolve().parents[2] / "hcu_adapter/deepep_dispatch.py"
spec = importlib.util.spec_from_file_location("dual_dispatch", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DualDispatchTests(unittest.TestCase):
    def setUp(self):
        self.vendor = Mock(return_value=object())
        self.upstream = Mock(return_value=object())
        self.dispatch = module.preserve_vendor_dispatch(self.vendor, self.upstream)
        self.buffer, self.x, self.ids, self.weights = (object() for _ in range(4))

    def test_vendor_int8_preserves_arguments(self):
        scales = object()
        result = self.dispatch(self.buffer, self.x, self.ids, self.weights, 96, 256,
                               quant_type=1, x_scales=scales, async_finish=True)
        self.assertIs(result, self.vendor.return_value)
        self.vendor.assert_called_once_with(self.buffer, self.x, self.ids, self.weights,
                                            96, 256, quant_type=1, x_scales=scales,
                                            async_finish=True)
        self.upstream.assert_not_called()

    def test_vendor_default_quantization(self):
        self.dispatch(self.buffer, self.x, self.ids, self.weights, 96, 256)
        self.vendor.assert_called_once_with(self.buffer, self.x, self.ids, self.weights, 96, 256)

    def test_vendor_keyword_weights(self):
        self.dispatch(self.buffer, self.x, self.ids, topk_weight=self.weights,
                      num_max_dispatch_tokens_per_rank=96, num_experts=256)
        self.assertEqual(self.vendor.call_count, 1)
        self.upstream.assert_not_called()

    def test_upstream_fp8_translation_retained(self):
        result = self.dispatch(self.buffer, self.x, self.ids, 96, 256,
                               use_fp8=True, topk_weights=self.weights)
        self.assertIs(result, self.upstream.return_value)
        self.upstream.assert_called_once_with(self.buffer, self.x, self.ids, 96, 256,
                                              use_fp8=True, topk_weights=self.weights)
        self.vendor.assert_not_called()

    def test_original_caller_routing_weights_retained(self):
        topk_weights = self.weights
        self.dispatch(self.buffer, self.x, self.ids, 96, 256)
        self.assertIs(self.upstream.call_args.kwargs["topk_weights"], topk_weights)

    def test_signature_advertises_both_contracts(self):
        params = inspect.signature(self.dispatch).parameters
        self.assertIn("quant_type", params)
        self.assertIn("use_fp8", params)

    def test_conflicting_modes_rejected(self):
        with self.assertRaises(TypeError):
            self.dispatch(self.buffer, self.x, self.ids, self.weights, 96, 256,
                          quant_type=1, use_fp8=True)
        self.vendor.assert_not_called()
        self.upstream.assert_not_called()

    def test_vendor_failure_does_not_fallback(self):
        self.vendor.side_effect = RuntimeError("vendor failed")
        with self.assertRaisesRegex(RuntimeError, "vendor failed"):
            self.dispatch(self.buffer, self.x, self.ids, self.weights, 96, 256, quant_type=1)
        self.upstream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
