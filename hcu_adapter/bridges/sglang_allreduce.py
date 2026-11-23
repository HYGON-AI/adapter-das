# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from functools import wraps

from hcu_adapter import registry


def install_ordered_tp_reduce():
    import torch
    from sglang.srt.distributed import parallel_state

    from hcu_adapter.kernels.ordered_tp_sum_triton import sum_bf16_tp8

    coordinator = parallel_state.GroupCoordinator
    original = coordinator._all_reduce_in_place
    if torch.version.hip is None or getattr(original, "_hcu_ordered_tp_reduce", False):
        return

    @wraps(original)
    def ordered_tp_reduce(self, input_) -> None:
        """Mutate input in place. The public all_reduce returns the tensor."""
        if (
            not input_.is_cuda
            or input_.dtype != torch.bfloat16
            or self.world_size != 8
            or self not in (parallel_state.get_tp_group(), parallel_state.get_attn_tp_group())
        ):
            return original(self, input_)
        if input_.numel() == 0:
            return

        flat_input = input_.contiguous().view(-1)
        parts = torch.empty(
            (self.world_size, flat_input.numel()),
            dtype=input_.dtype,
            device=input_.device,
        )
        self._all_gather_into_tensor(parts.view(-1), flat_input)
        sum_bf16_tp8(parts, flat_input)
        if not input_.is_contiguous():
            input_.copy_(flat_input.view(input_.shape))

    registry.register(
        "allreduce", ordered_tp_reduce,
        target=f"{parallel_state.__name__}:GroupCoordinator._all_reduce_in_place",
        mechanism=registry.MECH_SEAM,
        detail="8-rank BF16 TP all-gather and fixed rank order sum",
    )
    ordered_tp_reduce._hcu_ordered_tp_reduce = True
    coordinator._all_reduce_in_place = ordered_tp_reduce
