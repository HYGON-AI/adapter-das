# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""weak_ref_tensor — breakable-graph capture support (behavior restoration).

Upstream sglang's `compilation.weak_ref_tensor` imports this symbol from
sgl_kernel: a custom C++ TensorImpl that does NOT hold a storage reference,
so breakable-graph segments only pin their intermediates via the mempool
use-count (blocks stay reserved for replay addresses) instead of via
Python refs. That decoupling is what lets later capture buckets REUSE the
shared graph pool's blocks — without it every bucket's transient peak stays
resident and the 58-bucket prefill table OOMs partway through capture
(diagnosed 2026-10-14: 0.6B died at bucket 25/58 with each of the first
buckets holding 1.7-6 GB resident).

Prefer the vendor C++ op. Without it, use Torch's non-owning storage and
tensor constructors, as in its cudagraph-tree implementation. Admission
checks only the two entry points and does not initialize a device. The
view_as fallback remains for Torch builds without those entry points.
"""

import torch


def _load_vendor_impl():
    """Prefer the compiled C++ op (storage-decoupled TensorImpl).

    A vendor sgl_kernel wheel (DTK ships one) may be shadowed by OUR package
    of the same name. Its C++ op registers torch.ops.sgl_kernel.weak_ref_
    tensor from common_ops*.so, but the .so is a pybind module whose
    PyInit name is fixed ("common_ops") — load it under exactly that name
    via importlib with the vendor file location, then read the registered
    torch.ops symbol. Never import the vendor package __init__ (same-name
    shadowing would re-enter this file).
    """
    import glob
    import importlib
    import importlib.util
    import os
    import sys

    import torch

    if "sgl_kernel.common_ops" in sys.modules:
        mod = sys.modules["sgl_kernel.common_ops"]
    else:
        here_pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        so_candidates = []
        for entry in sys.path:
            if not entry or not os.path.isabs(entry):
                continue
            pkg = os.path.join(entry, "sgl_kernel")
            if os.path.normpath(pkg) == os.path.normpath(here_pkg):
                continue  # our own tree — skip
            so_candidates += glob.glob(os.path.join(pkg, "common_ops*.so"))
        if not so_candidates:
            return None
        so = sorted(so_candidates)[0]
        spec = importlib.util.spec_from_file_location(
            "sgl_kernel.common_ops", so
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["sgl_kernel.common_ops"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop("sgl_kernel.common_ops", None)
            return None

    # Inspect registration without allocating a CUDA test tensor. The vendor
    # op has no CPU kernel, so a CPU tensor probe would silently lose the
    # storage-decoupled CUDA implementation and retain graph intermediates.
    fn = None
    try:
        candidate = torch.ops.sgl_kernel.weak_ref_tensor
        if any(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "sgl_kernel::weak_ref_tensor", key
            )
            for key in ("CUDA", "CompositeExplicitAutograd", "CompositeImplicitAutograd")
        ):
            fn = candidate
    except (AttributeError, RuntimeError):
        pass
    if fn is None:
        fn = getattr(mod, "weak_ref_tensor", None)
    if not callable(fn):
        return None

    def _impl(t):
        return fn(t) if isinstance(t, torch.Tensor) else t

    return _impl


_vendor_weak_ref_tensor = _load_vendor_impl()


def _torch_weak_ref_impl(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"weak_ref_tensor expects a Tensor, got {type(t)!r}")
    if t.numel() == 0:
        return t
    if t.device.type != "cuda":
        return t.view_as(t)
    span = 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    storage = torch._C._construct_storage_from_data_pointer(
        t.data_ptr(), t.device, span * t.element_size()
    )
    metadata = {
        "size": t.shape,
        "stride": t.stride(),
        "dtype": t.dtype,
        "device": t.device,
        "storage_offset": 0,
    }
    return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)


_torch_weak_ref_tensor = (
    _torch_weak_ref_impl
    if hasattr(torch._C, "_construct_storage_from_data_pointer")
    and hasattr(torch._C, "_construct_CUDA_Tensor_From_Storage_And_Metadata")
    else None
)


if _vendor_weak_ref_tensor is not None:
    weak_ref_tensor = _vendor_weak_ref_tensor
elif _torch_weak_ref_tensor is not None:
    weak_ref_tensor = _torch_weak_ref_tensor
else:

    def weak_ref_tensor(t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"weak_ref_tensor expects a Tensor, got {type(t)!r}")
        if t.numel() == 0:
            return t
        # Strong-reference stand-in: view with identical shape/stride/layout.
        # (The upstream C++ op returns a TensorImpl sharing the Storage without
        # incrementing its refcount; nothing in torch core can express that, and
        # faking it via storage surgery risks use-after-free inside capture.)
        return t.view_as(t)


def weak_ref_tensor_lazy(t):  # keep signature parity for docs
    return weak_ref_tensor(t)


def weak_ref_tensors(tensors):
    """Mirror upstream convenience wrapper (tensor / list / tuple)."""
    import torch as _t

    if isinstance(tensors, _t.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError(
        "weak_ref_tensors expects a Tensor or list/tuple of Tensors"
    )
