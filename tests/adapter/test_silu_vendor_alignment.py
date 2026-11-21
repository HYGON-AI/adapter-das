# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU dispatch contracts; metadata doubles are not GPU numerical evidence.

Run with GPUs hidden: HCU_ADAPTER_HOST=none python -m pytest -q
tests/adapter/test_silu_vendor_alignment.py.
"""
from types import SimpleNamespace

import pytest
import torch

from sgl_kernel import elementwise


@pytest.mark.parametrize("backend", ["lightop", "aiter"])
@pytest.mark.parametrize("dtype,width", [
    (torch.float32, 6), (torch.float16, 12), (torch.bfloat16, 10),
])
@pytest.mark.parametrize("with_out", [False, True])
def test_invalid_alignment_precedes_vendor_probe_and_output_write(
        monkeypatch, backend, dtype, width, with_out):
    # Only the host-visible metadata is needed for the pre-launch contract.
    # No real GPU identity, tensor dispatch or provider numerics are simulated.
    x = SimpleNamespace(shape=(3, width), dtype=dtype, is_cuda=True,
                        device=torch.device("cpu"))
    out = torch.full((3, width // 2), -17, dtype=dtype) if with_out else None
    before = out.clone() if out is not None else None
    calls = []

    def resolve():
        calls.append("probe")
        return backend

    def lightop(input, output):
        calls.append("vendor")
        output.fill_(3)
        return output

    def aiter(output, input):
        return lightop(input, output)

    monkeypatch.setattr(elementwise, "_resolve_impl", resolve)
    monkeypatch.setattr(elementwise, "_LIGHTOP_IMPL", lightop)
    monkeypatch.setattr(elementwise, "_AITER_IMPL", aiter)
    with pytest.raises(ValueError, match="16 bytes"):
        elementwise.silu_and_mul(x, out)
    assert calls == []
    if out is not None:
        assert torch.equal(out, before)


@pytest.mark.parametrize("backend", ["lightop", "aiter"])
@pytest.mark.parametrize("dtype,width", [
    (torch.float32, 4), (torch.float16, 8), (torch.bfloat16, 8),
])
def test_exact_boundary_retains_vendor_call_and_out_identity(
        monkeypatch, backend, dtype, width):
    x = SimpleNamespace(shape=(3, width), dtype=dtype, is_cuda=True,
                        device=torch.device("cpu"))
    out = torch.full((3, width // 2), -17, dtype=dtype)
    calls = []

    def lightop(input, output):
        calls.append((input, output))

    def aiter(output, input):
        return lightop(input, output)

    monkeypatch.setattr(elementwise, "_resolve_impl", lambda: backend)
    monkeypatch.setattr(elementwise, "_LIGHTOP_IMPL", lightop)
    monkeypatch.setattr(elementwise, "_AITER_IMPL", aiter)
    assert elementwise.silu_and_mul(x, out) is out
    assert len(calls) == 1 and calls[0][0] is x and calls[0][1] is out
    assert torch.equal(out, torch.full_like(out, -17))
