# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Real public unquantized expert provider, raw weights and changed-input Graph.

Keep this in the fresh-interpreter vLLM host lane, outside common/SGLang tests.
"""

import pytest
import torch


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires HCU"
)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("activation_name", ("SILU", "GELU"))
@pytest.mark.parametrize("geometry", ((8, 256, 128, 2), (256, 2048, 512, 8)))
@torch.inference_mode()
def test_raw_experts_eager_and_changed_input_graph(
    monkeypatch, dtype, activation_name, geometry
):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    from hcu_adapter.bridges.vllm_moe_aiter import get_compatibility
    from hcu_adapter.bridges.vllm import SUPPORTED_HCU_ARCHES

    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in SUPPORTED_HCU_ARCHES:
        pytest.skip("requires a supported HCU platform")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")
    experts, hidden, intermediate, topk = geometry
    activation = getattr(MoEActivation, activation_name)
    config = FusedMoEConfig(
        num_experts=experts, experts_per_token=topk, hidden_dim=hidden,
        intermediate_size=intermediate, num_local_experts=experts,
        num_logical_experts=experts, activation=activation, device="cuda",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=dtype, moe_backend="aiter",
    )
    provider = get_compatibility()
    assert provider.architecture == arch
    assert provider.supports_config(config) == (True, None)
    provider.prepare()
    torch.manual_seed(931)
    rows = 3
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    w1 = torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=dtype)
    w1.mul_(hidden ** -.5)
    w2 = torch.randn(experts, hidden, intermediate, device="cuda", dtype=dtype)
    w2.mul_(intermediate ** -.5)
    original_w1, original_w2 = w1.clone(), w2.clone()
    ids = torch.empty(rows, topk, device="cuda", dtype=torch.int32)
    weights = torch.empty(rows, topk, device="cuda", dtype=torch.float32)
    backing = torch.full((rows + 2, hidden), 99., device="cuda", dtype=dtype)
    out = backing[1:-1]

    def produce(wave):
        x.copy_(torch.randn_like(x))
        ids.copy_((torch.arange(rows * topk, device="cuda").reshape(rows, topk)
                   * (wave + 1) + wave) % experts)
        weights.copy_(torch.randn_like(weights).softmax(-1))
        # Independent CPU FP32 reference for the selected raw experts, including
        # dtype storage between the two GEMMs. No vendor sorting/layout reuse.
        cpu_x, cpu_ids, cpu_weights = x.cpu().float(), ids.cpu(), weights.cpu()
        reference = torch.zeros(rows, hidden)
        for row in range(rows):
            for slot in range(topk):
                expert = int(cpu_ids[row, slot])
                gate, up = (w1[expert].cpu().float() @ cpu_x[row]).chunk(2)
                gate = (torch.nn.functional.silu(gate) if activation_name == "SILU"
                        else torch.nn.functional.gelu(gate))
                intermediate_value = (gate * up).to(dtype).float()
                down = (w2[expert].cpu().float() @ intermediate_value).to(dtype).float()
                reference[row].add_(down * cpu_weights[row, slot])
        return reference

    def run():
        provider.apply(config, FUSED_MOE_UNQUANTIZED_CONFIG, out, x, w1, w2,
            weights, ids, activation, experts, None, None, None, None, None,
            None, False)

    def check(reference):
        actual = out.cpu().float()
        error = (actual - reference).norm() / reference.norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0)
        assert error < .02, float(error)
        assert cosine > .999, float(cosine)
        torch.testing.assert_close(actual, reference, atol=.03, rtol=.03)
        assert bool((backing[[0, -1]] == 99).all())

    expected = produce(0)
    converted_w1, converted_w2 = provider.convert_weights(config, w1, w2)
    assert converted_w1 is w1 and converted_w2 is w2
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    check(expected)
    addresses = [t.data_ptr() for t in (x, w1, w2, ids, weights, out)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for wave in (1, 2):
        expected = produce(wave)
        graph.replay()
        torch.cuda.synchronize()
        observed = out.clone()
        check(expected)
        run()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, observed, atol=0, rtol=0)
        assert addresses == [t.data_ptr() for t in (x, w1, w2, ids, weights, out)]
    torch.testing.assert_close(w1, original_w1, atol=0, rtol=0)
    torch.testing.assert_close(w2, original_w2, atol=0, rtol=0)
