import sys


def preserve_vendor_dispatch(vendor_dispatch, upstream_dispatch):
    missing = object()

    def dispatch(self, x, topk_idx, *args, quant_type=missing, use_fp8=False, **kwargs):
        if quant_type is not missing or len(args) >= 3 or "topk_weight" in kwargs:
            if use_fp8:
                raise TypeError("Cannot combine vendor dispatch ABI with use_fp8=True")
            if quant_type is not missing:
                kwargs["quant_type"] = quant_type
            return vendor_dispatch(self, x, topk_idx, *args, **kwargs)
        if kwargs.get("topk_weights") is None:
            kwargs["topk_weights"] = sys._getframe(1).f_locals.get("topk_weights")
        return upstream_dispatch(self, x, topk_idx, *args, use_fp8=use_fp8, **kwargs)

    return dispatch
