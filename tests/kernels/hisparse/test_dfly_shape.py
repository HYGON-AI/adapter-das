"""Check the installed HIP resolver with wide DFly-style query rows."""

import pytest

from test_native import _equal, _gpu, _hot, _pinned, _resolve_reference, native, torch


@pytest.mark.parametrize('query_rows,region_stride', [(1, 4096), (6, 4096), (6, 14336)])
@pytest.mark.parametrize('graph_replay', [False, True])
def test_wide_draft_rows_pack_after_explicit_and_null_page_holes(
    native, query_rows, region_stride, graph_replay,
):
    page_size = 64
    row_width = 656
    topk = 2176
    disjoint = region_stride == 14336
    source_pages = list(range(1, 211 if disjoint else 36))
    source_pages[21] = 0
    resident_pages = [1] + [0] * (len(source_pages) - 1)
    hot_pages = list(range(2, 2 + region_stride // page_size))
    host_owner, host = _pinned((len(source_pages) + 1) * page_size, row_width, offset=19)
    hot_owner, hot, expected_storage, expected_hot = _hot(len(hot_pages) + 3, page_size, row_width)
    expected_hot[1].copy_(host[page_size:2 * page_size])
    hot.copy_(expected_hot.to(device='cuda'))

    logical_rows = []
    for index in range(query_rows):
        if disjoint:
            row = list(range(index * topk, (index + 1) * topk))
        else:
            row = list(range(2048)) + [-1] * (123 - index) + list(range(2048, 2053 + index))
        row[1023 + index] = -1
        assert len(row) == topk
        logical_rows.append(row)
    requests = [0] * query_rows
    states = [0]
    global_rows = [[-1] * region_stride]
    lru = [list(range(region_stride))]
    stats_ref = [0, 0]
    mapped = _gpu(global_rows)
    slots = _gpu(lru, torch.int16)
    stats = torch.zeros(2, dtype=torch.uint64, device='cuda')
    outputs = {key: torch.full((query_rows, topk), -77, dtype=torch.int32, device='cuda')
               for key in ('hot', 'mask', 'resolved', 'sh', 'sd', 'attention')}
    counts = torch.full((query_rows,), -77, dtype=torch.int32, device='cuda')
    copies = torch.full_like(counts, -77)
    logical_gpu = _gpu(logical_rows)
    hot_table = _gpu([hot_pages])
    state_map = _gpu(states)
    request_ids = _gpu(requests)
    source_table = _gpu([source_pages])
    resident_table = _gpu([resident_pages])

    def resolve_and_copy():
        native.hisparse_resolve_residency(
            host, hot, hot_table, logical_gpu, outputs['hot'], mapped, slots,
            state_map, region_stride, query_rows * topk if disjoint else region_stride,
            outputs['mask'], stats,
            outputs['attention'], page_size, request_ids, source_table, page_size,
            outputs['resolved'], counts, outputs['sh'], outputs['sd'], copies,
            resident_table, page_size, 0,
        )
        native.hisparse_gather_compact(host, hot, outputs['sh'], outputs['sd'], copies)

    graph = None
    if graph_replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            resolve_and_copy()
        torch.cuda.current_stream().wait_stream(stream)
        mapped.fill_(-1)
        slots.copy_(_gpu(lru, torch.int16))
        stats.zero_()
        hot.copy_(expected_hot.to(device='cuda'))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            resolve_and_copy()

    tail = query_rows * topk if disjoint else 2064
    changed_rows = [row[:-8] + list(range(tail, tail + 8)) for row in logical_rows]
    for rows in (logical_rows, changed_rows):
        logical_gpu.copy_(_gpu(rows))
        expected = _resolve_reference(
            rows, [hot_pages], global_rows, lru, host_rows=host.shape[0],
            requests=requests, states=states, source=[source_pages], resident=[resident_pages],
            block_size=page_size, attention_stride=page_size, stats=stats_ref,
        )
        if graph is None:
            resolve_and_copy()
        else:
            graph.replay()
        torch.cuda.synchronize()
        assert 0 < sum(expected['swaps']) <= region_stride
        assert all(1024 < count < topk for count in expected['counts'])
        for key in ('hot', 'mask', 'resolved', 'attention'):
            _equal(outputs[key], expected[key])
        _equal(counts, expected['counts'])
        _equal(copies, expected['swaps'])
        _equal(mapped, global_rows)
        _equal(slots, lru)
        _equal(stats, stats_ref)
        for row, copy_count in enumerate(expected['swaps']):
            # The consumer reads only this prefix; old copy-plan tails are scratch.
            for key in ('sh', 'sd'):
                _equal(outputs[key][row, :copy_count], expected[key][row][:copy_count])
            for index in range(copy_count):
                page, offset = divmod(expected['sd'][row][index], page_size)
                expected_hot[page, offset].copy_(host[expected['sh'][row][index]])
        _equal(hot_owner, expected_storage)
    assert host_owner.is_pinned()
