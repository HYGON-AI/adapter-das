# SPDX-License-Identifier: Apache-2.0
"""Native SHUFFLE scorer against an independent CPU reference and changed Graph replay."""
import hashlib
from pathlib import Path

import pytest
import torch

from hcu_adapter.kernels.shuffle_mqa_logits_triton import paged_logits


def inputs(seed, page_size, heads):
    generator = torch.Generator().manual_seed(seed)
    dim, pages, rows = 128, 5, 2
    logical = torch.randint(-3, 4, (pages, page_size, dim), generator=generator)
    scales = torch.randint(1, 5, (pages, page_size), generator=generator).float() / 4
    query = torch.randint(-3, 4, (rows, heads, dim), generator=generator)
    weights = torch.randint(1, 5, (rows, heads), generator=generator).float() / 4
    raw = torch.full((pages, page_size * (dim + 4)), 0xA5, dtype=torch.uint8)
    columns = torch.arange(dim)
    for slot in range(page_size):
        offsets = columns if page_size == 1 else (
            (slot // 16) * dim * 16 + (columns // 16) * 256
            + (slot % 16) * 16 + columns % 16)
        raw[:, offsets] = logical[:, slot].to(torch.float8_e4m3fn).view(torch.uint8)
    raw[:, page_size * dim:] = scales.contiguous().view(torch.uint8).reshape(pages, -1)
    width = 2 * page_size
    lengths = torch.tensor([seed % (width + 1), (seed * 7 + 1) % (width + 1)], dtype=torch.int32)
    tables = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    expected = torch.full((rows, width), -float('inf'))
    # Independent slot/page addressing and integer-valued FP64 dot products.
    for row in range(rows):
        for position in range(int(lengths[row])):
            page, slot = int(tables[row, position // page_size]), position % page_size
            dots = (query[row].double() * logical[page, slot].double()).sum(-1)
            expected[row, position] = (dots.relu() * weights[row].double()).sum() * scales[page, slot]
    arguments = (query.to(torch.float8_e4m3fn), raw.view(pages, page_size, dim + 4),
                 weights, lengths, tables)
    return arguments, expected, width


@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize("heads", [2, 64])
def test_native_scores_and_changed_graph(page_size, heads, request):
    if not torch.cuda.is_available():
        pytest.skip("requires an actual GPU")
    module = __import__(paged_logits.__module__, fromlist=['__file__'])
    source = Path(module.__file__)
    request.node.user_properties.extend([
        ('architecture', torch.cuda.get_device_properties(0).gcnArchName),
        ('kernel_source', str(source)),
        ('kernel_sha256', hashlib.sha256(source.read_bytes()).hexdigest()),
    ])
    initial, _, width = inputs(0, page_size, heads)
    device_inputs = tuple(value.cuda() for value in initial)
    addresses = [value.data_ptr() for value in device_inputs]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_logits(*device_inputs, width)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = paged_logits(*device_inputs, width)
    # Capture does not execute; replay the original zero/one-length state first.
    for seed in (0, 1, width - 1, width, 9, 31, 127):
        values, expected, _ = inputs(seed, page_size, heads)
        for destination, value in zip(device_inputs, values):
            destination.copy_(value)
        unchanged = [value.clone() for value in device_inputs]
        eager = paged_logits(*device_inputs, width)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(eager.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(captured.cpu(), expected, rtol=0, atol=0)
        for actual, original in zip(device_inputs, unchanged):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        assert addresses == [value.data_ptr() for value in device_inputs]
