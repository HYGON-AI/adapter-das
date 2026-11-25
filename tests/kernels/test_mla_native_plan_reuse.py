# SPDX-License-Identifier: Apache-2.0
"""Native gfx938 parity for step-local H64 plan reuse and Graph replay."""
from collections import Counter
from dataclasses import dataclass
import math
import os
from types import SimpleNamespace

import pytest
import torch

from hcu_adapter.kernels import mla_native_plan_triton
from hcu_adapter.kernels.mla_native_plan_reuse import NativePlanCache
from hcu_adapter.kernels.mla_sparse_decode_dtk import try_decode


ROWS = 120
HEADS = 64
DIM = 512
GROUP_SIZES = (6, 6, 6, 4, 4, 4, 4, 4)
PAGES = (32, 32, 32, 64, 64, 64, 64, 64)
ENTERING = ('full', 'short', 'mixed', 'full', 'short', 'empty', 'mixed', 'empty')
GUARD = 12345
POISON = -9999


def test_cache_keys_ordered_sources_and_split_storage(monkeypatch):
    stream = SimpleNamespace(cuda_stream=11)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: stream)
    seen = []
    monkeypatch.setattr(mla_native_plan_triton, 'refresh_guarded_plan',
                        lambda *args: seen.append(args))
    q = torch.empty(ROWS, HEADS, DIM, dtype=torch.bfloat16)
    positions = torch.arange(ROWS)
    swa = torch.empty(8, 32, 584, dtype=torch.uint8)
    c1a = torch.empty(4, 64, 584, dtype=torch.uint8)
    c2a = torch.empty(8, 32, 584, dtype=torch.uint8)
    extra = (torch.empty(ROWS * 512, dtype=torch.int32),
             torch.arange(ROWS + 1, dtype=torch.int32) * 512)
    lengths = (torch.full((ROWS,), 128, dtype=torch.int32),
               torch.full((ROWS,), 512, dtype=torch.int32))
    cache = NativePlanCache()
    first = cache.key(q, positions, swa, c1a, extra, lengths)
    assert first is not None
    meta = SimpleNamespace(have_initialized=True,
        tile_scheduler_metadata=torch.empty(64, 8, dtype=torch.int32),
        num_splits=torch.arange(ROWS + 1, dtype=torch.int32))
    cache.remember(first, lengths, meta)
    changed = (lengths[0].clone(), lengths[1].clone())
    changed[0][4] = 1
    assert cache.lookup(cache.key(q, positions[:], swa, c1a, extra, changed), changed) is meta
    assert seen[-1][2] is lengths
    cache.remember(first, changed, meta)
    assert cache.lookup(first, lengths) is meta
    assert seen[-1][2] is changed
    assert cache.lookup(cache.key(q, positions.flip(0), swa, c1a, extra, lengths), lengths) is None
    assert cache.lookup(cache.key(q, positions, swa, c2a, extra, lengths), lengths) is None
    assert cache.lookup(cache.key(q, positions, swa, c1a, (extra[0], extra[1].clone()), lengths), lengths) is None
    stream.cuda_stream = 12
    assert cache.lookup(cache.key(q, positions, swa, c1a, extra, lengths), lengths) is None
    assert NativePlanCache().lookup(first, lengths) is None
    meta.num_splits = meta.num_splits.clone()
    with pytest.raises(AssertionError):
        cache.lookup(first, lengths)


@pytest.fixture(scope='module')
def native():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip('requires DTK and a real gfx938')
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(':')[0] != 'gfx938' or props.multi_processor_count != 64:
        pytest.skip('requires a 64 CU gfx938 device')
    import vllm._C_stable_libtorch  # noqa: F401
    from flash_mla import flash_mla_interface

    assert mla_native_plan_triton.qualify_native_plan('cuda:0', guarded_reuse=True)
    return flash_mla_interface


def packed_cache(page, seed):
    # Match the model's page-flat D512/OCP 584-byte record and pitched pages.
    pages = 1024 // page
    generator = torch.Generator().manual_seed(seed)
    values = (torch.randn(pages, page, DIM, generator=generator) * .2).bfloat16()
    data = torch.cat((values[..., :448].to(torch.float8_e4m3fn).view(torch.uint8),
                      values[..., 448:].contiguous().view(torch.uint8)), dim=-1)
    scales = torch.full((pages, page * 8), 127, dtype=torch.uint8)
    packed = torch.cat((data.reshape(pages, -1), scales), dim=-1)
    stride = 105984
    cache = torch.empty((pages, stride), dtype=torch.uint8, device='cuda')
    view = cache[:, :page * 584].view(pages, page, 584)
    view.copy_(packed.view(pages, page, 584))
    return view


def csr(width):
    return (torch.empty(ROWS * width, dtype=torch.int32, device='cuda'),
            torch.empty(ROWS + 1, dtype=torch.int32, device='cuda'))


def set_bank(bank, lengths, seed):
    flat, indptr = bank
    width = flat.numel() // ROWS
    padded = torch.full((flat.numel(),), -1, dtype=torch.int32)
    offsets = [0]
    for row, length in enumerate(lengths):
        start = offsets[-1]
        end = start + length
        padded[start:end] = (torch.arange(length) + 11 * row + seed) % 1024
        if row == 1 and 0 < length < width:
            padded[end - 1] = 1024 + 17  # Must be removed from the ordered count.
        offsets.append(end)
    flat.copy_(padded)
    indptr.copy_(torch.tensor(offsets, dtype=torch.int32))


def row_lengths(kind):
    if kind == 'full':
        return [128] * ROWS, [512] * ROWS
    if kind == 'short':
        return [128] * ROWS, [64] * ROWS
    if kind == 'empty':
        return [0, 128, 128, 128, 128, 128] * 20, [0, 512, 512, 512, 512, 512] * 20
    assert kind == 'mixed'
    return [0, 1, 63, 64, 128, 128] * 20, [0, 1, 64, 256, 512, 64] * 20


def alternate_main(kind, main):
    if kind == 'full':
        return [64] * ROWS
    if kind == 'short':
        return [64] + main[1:]
    swapped = main.copy()
    # Mixed keeps the same joint length histogram while moving a query across
    # a tile boundary. Empty introduces SWA-only and compressed-only rows.
    a, b = (2, 5) if kind == 'mixed' else (0, 2)
    swapped[a], swapped[b] = swapped[b], swapped[a]
    return swapped


@dataclass
class Call:
    builds: bool
    storage: tuple
    plan: torch.Tensor
    splits: torch.Tensor
    main: torch.Tensor
    extra: torch.Tensor


class CallTrace:
    def __init__(self, native):
        self.native = native
        self.records = []

    def __call__(self, **kwargs):
        metadata = kwargs['tile_scheduler_metadata']
        planned = metadata.tile_scheduler_metadata is None
        result = self.native.flash_mla_with_kvcache(**kwargs)
        plan, splits = metadata.tile_scheduler_metadata, metadata.num_splits
        # The slot can be overwritten by a later consumer. Capture values at
        # this call, after the guard and native compute, on the same stream.
        self.records.append(Call(planned, (plan, splits), plan.clone(), splits.clone(),
                                 kwargs['topk_length'], kwargs['extra_topk_length']))
        return result


class Workload:
    def __init__(self):
        self.main = [[csr(128) for _ in range(count)] for count in GROUP_SIZES]
        self.extra = [csr(512) for _ in GROUP_SIZES]
        self.swa = [packed_cache(32, 902), packed_cache(32, 903)]
        self.kv = {32: packed_cache(32, 904), 64: packed_cache(64, 905)}
        self.positions = torch.arange(ROWS, device='cuda', dtype=torch.int64)
        self.sink = torch.linspace(-.2, .2, HEADS, device='cuda', dtype=torch.float32)
        self.q = [torch.empty((ROWS, HEADS, DIM), device='cuda', dtype=torch.bfloat16)
                  for _ in range(2)]

    def update(self, kinds, seed, *, diverge):
        self.positions.copy_(torch.arange(seed, seed + ROWS, dtype=torch.int64))
        for group, kind in enumerate(kinds):
            main, extra = row_lengths(kind)
            alternate = alternate_main(kind, main)
            for offset, bank in enumerate(self.main[group]):
                lengths = alternate if diverge and offset in (1, 2, 5) else main
                set_bank(bank, lengths, seed + group + offset)
            set_bank(self.extra[group], extra, seed + 5 * group)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        for query in self.q:
            query.copy_(torch.randn(query.shape, generator=generator, device='cuda') * .2)
        torch.cuda.synchronize()

    def source_pointers(self):
        banks = [bank for group in self.main for bank in group] + self.extra
        return [tensor.data_ptr() for bank in banks for tensor in bank]

    def run(self, native, outputs, reuse):
        provider = CallTrace(native)
        cache = NativePlanCache() if reuse else None
        layer = 0
        for group, count in enumerate(GROUP_SIZES):
            extra_ids, extra_indptr = self.extra[group]
            for offset in range(count):
                swa_ids, swa_indptr = self.main[group][offset]
                head = layer % 2
                result = try_decode(q=self.q[head], kv_cache=self.kv[PAGES[group]],
                    swa_k_cache=self.swa[head], swa_only=False,
                    topk_indices=None, topk_lens=None, swa_indices=None, swa_lens=None,
                    swa_ragged_indices=swa_ids, swa_ragged_indptr=swa_indptr,
                    topk_ragged_indices=extra_ids, topk_ragged_indptr=extra_indptr,
                    attn_sink=self.sink, scale=1 / math.sqrt(DIM),
                    head_dim=DIM, nope_head_dim=448, rope_head_dim=64,
                    output=outputs[layer, :ROWS], inv_rope_positions=self.positions,
                    provider=provider, metadata_factory=native.get_mla_metadata,
                    epilogue=None, plan_cache=cache)
                assert result == 0
                layer += 1
        return provider.records


def output_storage():
    return torch.full((sum(GROUP_SIZES), ROWS + 2, HEADS, DIM), GUARD,
                      device='cuda', dtype=torch.bfloat16)


def assert_outputs(candidate, oracle, layers):
    assert torch.equal(candidate, oracle)
    assert torch.all(candidate[:, ROWS:] == GUARD)
    for layer, record in enumerate(layers):
        live = record.main + record.extra > 0
        assert torch.isfinite(candidate[layer, :ROWS]).all()
        assert torch.count_nonzero(candidate[layer, :ROWS][live]) > 0
        assert torch.count_nonzero(candidate[layer, :ROWS][~live]) == 0


def assert_plans(candidate, oracle):
    assert len(candidate) == len(oracle) == sum(GROUP_SIZES)
    assert all(record.builds for record in oracle)
    group_starts = [sum(GROUP_SIZES[:i]) for i in range(len(GROUP_SIZES))]
    assert [i for i, record in enumerate(candidate) if record.builds] == group_starts
    for group, start in enumerate(group_starts):
        shared = tuple(x.data_ptr() for x in candidate[start].storage)
        for layer in range(start, start + GROUP_SIZES[group]):
            actual, expected = candidate[layer], oracle[layer]
            assert torch.equal(actual.main, expected.main) and torch.equal(actual.extra, expected.extra)
            assert tuple(x.data_ptr() for x in actual.storage) == shared
            assert torch.equal(actual.splits, expected.splits), (group, layer)
            inactive = expected.plan[:, 0] >= ROWS
            # H64 ignores dword 7 and exits inactive parts before using any
            # start/end tiles. DTK's inactive begin tile can differ between
            # native invocations. All active fields and row sentinels are exact.
            assert torch.equal(actual.plan[:, :2], expected.plan[:, :2]), (group, layer)
            assert torch.equal(actual.plan[~inactive, :7], expected.plan[~inactive, :7]), (group, layer)
            assert torch.all(actual.plan[inactive, 0] == ROWS)
            assert torch.all(actual.plan[inactive, 1] == ROWS - 1)
            assert torch.all(actual.plan[:, :7] != POISON)
    assert len({candidate[i].storage[0].data_ptr() for i in group_starts}) == len(GROUP_SIZES)
    assert len({candidate[i].storage[1].data_ptr() for i in group_starts}) == len(GROUP_SIZES)


@torch.inference_mode()
def test_native_single_step_matches_fresh_plans(native):
    task = Workload()
    task.update(ENTERING, 901, diverge=True)
    candidate, oracle = output_storage(), output_storage()
    cached = task.run(native, candidate, reuse=True)
    fresh = task.run(native, oracle, reuse=False)
    assert_plans(cached, fresh)
    assert_outputs(candidate, oracle, fresh)
    same, changed = cached[12], cached[13]
    assert Counter(zip(same.main.tolist(), same.extra.tolist())) == Counter(
        zip(changed.main.tolist(), changed.extra.tolist()))
    assert not torch.equal(same.main, changed.main)
    assert not torch.equal(same.plan[:, :7], changed.plan[:, :7])
    print('native step: 8 native plans / 38 layers; both layouts, reordered lengths, splits, outputs')


@torch.inference_mode()
def test_guard_compares_both_length_vectors(native):
    main = torch.full((ROWS,), 128, dtype=torch.int32, device='cuda')
    extra = (torch.arange(ROWS, device='cuda', dtype=torch.int32) % 8 + 1) * 64
    plan = torch.full((64, 8), POISON, device='cuda', dtype=torch.int32)
    splits = torch.full((ROWS + 1,), POISON, device='cuda', dtype=torch.int32)
    refresh = mla_native_plan_triton.refresh_guarded_plan
    refresh(main, extra, (main.clone(), extra.clone()), plan, splits)
    assert torch.all(plan == POISON) and torch.all(splits == POISON)
    refresh(main, extra.roll(1), (main, extra), plan, splits)
    assert torch.all(plan != POISON) and torch.all(splits != POISON)
    assert int(splits[0]) == 0 and torch.all(splits.diff() >= 1)


@torch.inference_mode()
def test_native_graph_replay_rebuilds_group_plans(native):
    if os.getenv('HCU_MLA_PLAN_TEST_GRAPH') != '1':
        pytest.skip('run after the native single-step check has passed')
    task = Workload()
    task.update(ENTERING, 901, diverge=False)
    candidate, oracle = output_storage(), output_storage()
    task.run(native, candidate, reuse=True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cached = task.run(native, candidate, reuse=True)
    pointers = [tuple(x.data_ptr() for x in row.storage) for row in cached]
    source_ptrs = task.source_pointers()
    assert len(set(pointers)) == len(GROUP_SIZES)
    later = [ENTERING, ('mixed', 'empty', 'full', 'short', 'mixed', 'full', 'empty', 'short'),
             ('short', 'full', 'empty', 'mixed', 'full', 'short', 'full', 'mixed')]
    previous = None
    for index, kinds in enumerate(later):
        task.update(kinds, 1001 + index, diverge=index != 1)
        for row in cached:
            if row.builds:
                row.storage[0].fill_(POISON)
                row.storage[1].fill_(POISON)
        candidate.fill_(GUARD)
        graph.replay()
        fresh = task.run(native, oracle, reuse=False)
        assert_plans(cached, fresh)
        assert_outputs(candidate, oracle, fresh)
        assert [tuple(x.data_ptr() for x in row.storage) for row in cached] == pointers
        assert task.source_pointers() == source_ptrs
        newest = torch.cat([torch.cat((row.plan[:, :2].flatten(), row.splits)) for row in cached])
        if previous is not None:
            assert not torch.equal(previous, newest)
        previous = newest.clone()
    print('native Graph: 3 replays with distinct lengths and ordered divergence; 8 stable plan/split pairs')
