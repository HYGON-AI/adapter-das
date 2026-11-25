# SPDX-License-Identifier: Apache-2.0
"""Forward-local H64 scheduling plans for V4.1 compressed index sources."""
from dataclasses import dataclass

import torch


_PARTS = 64
_PLAN_FIELDS = 8
_ROWS = 120


def _view(tensor):
    return (tensor.device, tensor.dtype, tensor.data_ptr(), tuple(tensor.shape), tensor.stride())


def _layout(cache):
    return (cache.dtype, tuple(cache.shape), cache.stride())


@dataclass
class _Plan:
    metadata: object
    ordered_sources: tuple
    output_views: tuple
    last_lengths: tuple


class NativePlanCache:
    """Pair a native plan and split prefix with their last ordered lengths.

    Core memoizes the compressed CSR per index source and ratio. The SWA groups
    publish separate storage. On each replay the first consumer builds the plan
    with the native provider. Later consumers compare both complete post-filter
    length vectors in row order on the device. A mismatch rewrites the plan and
    splits before this consumer runs. Selector IDs always come from this layer.
    The stream in the key orders all reads and overwrites of the shared buffers.
    """

    def __init__(self):
        self._plans = {}

    def key(self, q, positions, main_cache, extra_cache, extra, lengths):
        if (tuple(q.shape) != (_ROWS, 64, 512)
                or main_cache.shape[1] != 32 or extra_cache.shape[1] not in (32, 64)
                or positions is None or positions.shape != (_ROWS,)
                or positions.device != q.device):
            return None
        if any(x is None or x.shape != (_ROWS,) or x.device != q.device
               or x.dtype != torch.int32 or not x.is_contiguous() for x in lengths):
            return None
        extra_values, extra_indptr = extra
        if (extra_values is None or extra_values.numel() != _ROWS * 512
                or extra_indptr is None or extra_indptr.shape != (_ROWS + 1,)):
            return None
        # Core publishes one compressed CSR per index source and ratio. Its
        # exact views identify that source, while positions anchors query row
        # order. Both page layouts must agree for the native metadata object.
        # SWA storage belongs to each KV group. The device guard checks its
        # effective lengths along with the compressed bank's on every hit.
        sources = (positions, extra_values, extra_indptr)
        key = (torch.cuda.current_stream(q.device).cuda_stream,
               _layout(main_cache), _layout(extra_cache), tuple(map(_view, sources)),
               (_PARTS, _PLAN_FIELDS, _ROWS + 1))
        return key, sources

    def lookup(self, token, lengths):
        key, sources = token
        entry = self._plans.get(key)
        if entry is None:
            return None
        assert tuple(map(_view, entry.ordered_sources)) == tuple(map(_view, sources)) == key[3]
        assert self._output_views(entry.metadata) == entry.output_views
        from .mla_native_plan_triton import refresh_guarded_plan

        meta = entry.metadata
        refresh_guarded_plan(*lengths, entry.last_lengths, meta.tile_scheduler_metadata, meta.num_splits)
        return meta

    def remember(self, token, lengths, metadata):
        key, sources = token
        entry = self._plans.get(key)
        if entry is not None:
            # The vendor can re-wrap tensors, but all graph nodes keep the same
            # native plan and prefix storage bound to their argument addresses.
            assert entry.metadata is metadata and self._output_views(metadata) == entry.output_views
            entry.last_lengths = lengths
            return
        plan = getattr(metadata, 'tile_scheduler_metadata', None)
        prefix = getattr(metadata, 'num_splits', None)
        parts, fields, split_words = key[-1]
        device = sources[0].device
        if (getattr(metadata, 'have_initialized', False)
                and isinstance(plan, torch.Tensor) and isinstance(prefix, torch.Tensor)
                and plan.shape == (parts, fields) and prefix.shape == (split_words,)
                and plan.device == prefix.device == device
                and plan.dtype == prefix.dtype == torch.int32
                and plan.is_contiguous() and prefix.is_contiguous()):
            self._plans[key] = _Plan(metadata, sources, self._output_views(metadata), lengths)

    @staticmethod
    def _output_views(metadata):
        return (_view(metadata.tile_scheduler_metadata), _view(metadata.num_splits))
