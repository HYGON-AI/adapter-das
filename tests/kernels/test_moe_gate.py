# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MoE gate contracts, including finite softmax ties on CPU and HCU/CUDA.

Run on a node with a working GPU: `pytest tests/kernels/test_moe_gate.py -v`.
Device-only cases are skipped automatically when no GPU is visible.
"""

import math

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU/CUDA device",
)


@requires_gpu
def test_topk_softmax_gpu_matches_cpu():
    torch.manual_seed(6)
    M, E, K = 64, 256, 8
    logits = (torch.randn(M, E, device="cuda") * 4).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)
    w = torch.empty(M, K, device="cuda", dtype=torch.float32)
    ids = torch.empty(M, K, device="cuda", dtype=torch.int32)
    sgl_kernel.topk_softmax(w, ids, logits, renormalize=True,
                            moe_softcapping=0.0, correction_bias=bias)

    lf = logits.float()
    act = torch.softmax(lf, dim=-1)
    _, rids = torch.topk(lf + bias, k=K, dim=-1, sorted=False)
    rw = act.gather(1, rids)
    rw = rw / rw.sum(-1, keepdim=True)
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-3, atol=1e-4
    )


@requires_gpu
def test_moe_fused_gate_gpu_grouped():
    torch.manual_seed(8)
    M, E, K, G, KG = 32, 64, 8, 8, 4
    logits = torch.randn(M, E, device="cuda")
    bias = torch.randn(E, device="cuda")
    w, ids = sgl_kernel.moe_fused_gate(
        logits, bias, topk=K, scoring_func="sigmoid", renormalize=True,
        num_expert_group=G, topk_group=KG,
    )
    assert w.shape == (M, K) and ids.shape == (M, K)
    scores = torch.sigmoid(logits)
    biased = scores + bias
    gs = biased.view(M, G, E // G).topk(2, dim=-1).values.sum(-1)
    keep = gs.topk(KG, dim=-1, sorted=False).indices
    gm = torch.zeros_like(gs, dtype=torch.bool).scatter(1, keep, True)
    em = gm.unsqueeze(-1).expand(M, G, E // G).reshape(M, E)
    masked = biased.masked_fill(~em, float("-inf"))
    _, rids = torch.topk(masked, k=K, dim=-1, sorted=False)
    rw = scores.gather(1, rids)
    rw = rw / rw.sum(-1, keepdim=True)
    torch.testing.assert_close(
        torch.sort(w, dim=-1).values, torch.sort(rw, dim=-1).values, rtol=1e-4, atol=1e-5
    )
    assert torch.equal(torch.sort(ids, dim=-1).values, torch.sort(rids, dim=-1).values)


def _tied_logits(dtype):
    """Finite router rows with ties both within and across the top-k boundary."""
    rows = torch.zeros(4, 256, dtype=dtype)
    view = rows[:, ::2]  # Exercise the public strided-input contract as well.
    view[1, 96:104] = 3
    view[1, 28] = view[1, 67] = 2
    view[2, 3:10] = 4
    view[2, 28] = view[2, 67] = 2
    view[3, ::3] = -1
    view[3, 19] = 2
    return view


def _softmax_tie_oracle(logits, renormalize):
    # Independent scalar ordering: descending logit, then lowest expert ID.
    # Both the community Triton router and power-of-two AOT CUDA kernel use
    # this tie rule; arbitrary torch.topk tie winners are not the oracle.
    expert_ids, weights = [], []
    for row in logits.float().cpu().tolist():
        selected = sorted(range(len(row)), key=lambda i: (-row[i], i))[:8]
        maximum = max(row)
        numerators = [math.exp(x - maximum) for x in row]
        denominator = math.fsum(
            numerators[i] for i in selected
        ) if renormalize else math.fsum(numerators)
        expert_ids.append(selected)
        weights.append([numerators[i] / denominator for i in selected])
    return torch.tensor(expert_ids, dtype=torch.int32), torch.tensor(weights)


def _softmax_gate(entry, logits, weights, ids, renormalize):
    if entry == "topk_softmax":
        sgl_kernel.topk_softmax(weights, ids, logits, renormalize=renormalize)
    else:
        result_weights, result_ids = sgl_kernel.moe_fused_gate(
            logits, None, topk=8, scoring_func="softmax",
            renormalize=renormalize,
        )
        weights.copy_(result_weights)
        ids.copy_(result_ids)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("entry", ["topk_softmax", "moe_fused_gate"])
def test_softmax_ties_choose_lowest_expert_id(device, dtype, renormalize, entry):
    cpu_logits = _tied_logits(dtype)
    storage = torch.empty(4, 256, device=device, dtype=dtype)
    logits = storage[:, ::2]
    logits.copy_(cpu_logits)
    weights = torch.empty(4, 8, device=device)
    ids = torch.empty(4, 8, device=device, dtype=torch.int32)
    expected_ids, expected_weights = _softmax_tie_oracle(cpu_logits, renormalize)
    for _ in range(3):
        _softmax_gate(entry, logits, weights, ids, renormalize)
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.cpu(), expected_weights, rtol=1e-6, atol=1e-7
        )


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("entry", ["topk_softmax", "moe_fused_gate"])
def test_softmax_ties_follow_changed_graph_inputs(dtype, entry):
    cpu_logits = _tied_logits(dtype)
    logits = cpu_logits.to("cuda")
    weights = torch.empty(4, 8, device="cuda")
    ids = torch.empty(4, 8, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _softmax_gate(entry, logits, weights, ids, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _softmax_gate(entry, logits, weights, ids, True)
    for shift in [0, 17, 49, 0]:
        changed = cpu_logits.roll(shift, dims=1)
        logits.copy_(changed)
        graph.replay()
        expected_ids, expected_weights = _softmax_tie_oracle(changed, True)
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.cpu(), expected_weights, rtol=1e-6, atol=1e-7
        )
