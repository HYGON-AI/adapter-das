import types

from hcu_adapter.small_copy_jit import adapt_kernel


def test_alias_identity_options_and_idempotence():
    calls = []
    class Standard:
        def __init__(self, fn, **kwargs):
            calls.append((fn, kwargs))
        def run(self, *args, **kwargs):
            return args, kwargs
    Fast = type("FastJITFunction", (), {"__module__": "triton.backends.hcu.jit"})
    kernel = Fast()
    kernel.fn = lambda: None
    kernel._init_kwargs = {"do_not_specialize": ["sizes"], "debug": False}
    alias = kernel
    assert adapt_kernel(kernel, Standard)
    assert alias is kernel
    assert alias.run(7, grid=(2,)) == ((7,), {"grid": (2,)})
    assert calls == [(kernel.fn, kernel._init_kwargs)]
    assert not adapt_kernel(kernel, Standard)
    assert len(calls) == 1


def test_other_jit_is_untouched():
    kernel = types.SimpleNamespace(run=object())
    original = kernel.run
    assert not adapt_kernel(kernel, None)
    assert kernel.run is original
