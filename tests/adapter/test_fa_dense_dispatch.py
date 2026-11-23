# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Dense/ragged routing contract; runs without a GPU or vendor wheel."""
import sys
import types

import pytest
import torch

from hcu_adapter.fa_vendor import _preserve_dense_varlen, _varlen_triton_ok


@pytest.mark.parametrize("keyword", [False, True])
def test_dense_preserves_batches_and_fa2_return(monkeypatch, keyword):
    q = torch.empty(2, 5, 4, 64)
    k = v = torch.empty(2, 3, 2, 64)
    result = (object(), object(), object())
    calls = []

    def dense(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_func=dense))

    @_preserve_dense_varlen
    def ragged(*args, **kwargs):
        pytest.fail("dense input reached ragged adapter")

    options = dict(softmax_scale=0.125, causal=True, window_size=(4, 0),
                   softcap=2.0, return_softmax_lse=True)
    if keyword:
        actual = ragged(q=q, k=k, v=v, cu_seqlens_q=None, cu_seqlens_k=None, **options)
    else:
        actual = ragged(q, k, v, None, None, **options)
    assert actual is result
    args, kwargs = calls.pop()
    assert args[0] is q and args[1] is k and args[2] is v
    assert kwargs == dict(softmax_scale=0.125, causal=True, window_size=(4, 0),
                          softcap=2.0, return_attn_probs=True)


def test_ragged_preserves_arguments():
    q = k = v = torch.empty(7, 2, 64)
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    calls = []
    token = object()

    @_preserve_dense_varlen
    def ragged(*args, **kwargs):
        calls.append((args, kwargs))
        return token

    assert ragged(q, k, v, cu, cu, out=token) is token
    args, kwargs = calls.pop()
    assert args[0] is q and args[3] is cu and args[4] is cu
    assert kwargs == {"out": token}


@pytest.mark.parametrize("boundary", ["neither", "q", "k"])
def test_incomplete_ragged_metadata_rejected(boundary):
    q = k = v = torch.empty(7, 2, 64)
    cu = torch.tensor([0, 7], dtype=torch.int32)
    p = dict(q=q, k=k, v=v, cu_seqlens_q=cu if boundary == "q" else None,
             cu_seqlens_k=cu if boundary == "k" else None)
    assert not _varlen_triton_ok(p)
    with pytest.raises(ValueError, match="both cu_seqlens"):
        _preserve_dense_varlen(lambda **kwargs: pytest.fail("invalid ragged admitted"))(**p)


@pytest.fixture
def fallback_host(monkeypatch):
    import sys
    import types
    from hcu_adapter import hcu_env, registry

    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.setattr(hcu_env, "resolve", lambda name: ("vendor", "explicit"))
    monkeypatch.setattr(hcu_env, "arch_name", lambda: "gfx936")
    monkeypatch.setattr(registry, "_RECORDS", {})
    monkeypatch.setattr(registry, "_BOUND", {})
    modules = {}
    for name in ("sglang", "sglang.srt", "sglang.srt.arg_groups",
                 "sglang.srt.arg_groups.overrides", "sglang.srt.arg_groups.attention_hook"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, attr = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], attr, module)
    overrides = modules["sglang.srt.arg_groups.overrides"]
    consumer = modules["sglang.srt.arg_groups.attention_hook"]

    def original(view):
        return {"fallback": (view.attention_backend, view.kv_cache_dtype)}

    overrides._attention_backend_fa3_fp8_fallback = original
    consumer._attention_backend_fa3_fp8_fallback = original
    overrides.POST_PROCESS_PASSES = [original, object(), original]
    return overrides, consumer, original


def test_e5m2_guard_updates_consumers_and_preserves_fallback(fallback_host):
    from types import SimpleNamespace
    from hcu_adapter import fa_vendor

    overrides, consumer, original = fallback_host
    fa_vendor._install_e5m2_backend_guard()
    installed = overrides._attention_backend_fa3_fp8_fallback
    assert installed is consumer._attention_backend_fa3_fp8_fallback
    assert overrides.POST_PROCESS_PASSES[0] is installed
    assert overrides.POST_PROCESS_PASSES[2] is installed
    assert installed(SimpleNamespace(attention_backend="fa3", kv_cache_dtype="fp8_e5m2")) == {}
    for backend, dtype in (("fa3", "fp8_e4m3"), ("triton", "fp8_e5m2")):
        view = SimpleNamespace(attention_backend=backend, kv_cache_dtype=dtype)
        assert installed(view) == original(view)
    fa_vendor._install_e5m2_backend_guard()
    assert overrides._attention_backend_fa3_fp8_fallback is installed


@pytest.mark.parametrize("mode,arch,hip", [
    ("auto", "gfx936", "test"), ("triton", "gfx936", "test"),
    ("emu", "gfx936", "test"), ("vendor", "gfx942", "test"),
    ("vendor", "gfx936", None),
])
def test_e5m2_guard_disabled_keeps_all_bindings(fallback_host, monkeypatch, mode, arch, hip):
    from hcu_adapter import fa_vendor, hcu_env, registry

    overrides, consumer, original = fallback_host
    monkeypatch.setattr(hcu_env, "resolve", lambda name: (mode, "explicit"))
    monkeypatch.setattr(hcu_env, "arch_name", lambda: arch)
    monkeypatch.setattr(torch.version, "hip", hip)
    fa_vendor._install_e5m2_backend_guard()
    assert overrides._attention_backend_fa3_fp8_fallback is original
    assert consumer._attention_backend_fa3_fp8_fallback is original
    assert overrides.POST_PROCESS_PASSES[0] is original
    assert registry.get("flash_attention") is None


def test_e5m2_guard_registration_failure_is_atomic(fallback_host, monkeypatch):
    from hcu_adapter import fa_vendor, registry

    overrides, consumer, original = fallback_host
    register = registry.register
    count = 0

    def reject_second(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("registration refused")
        return register(*args, **kwargs)

    monkeypatch.setattr(registry, "register", reject_second)
    with pytest.raises(ValueError, match="registration refused"):
        fa_vendor._install_e5m2_backend_guard()
    assert overrides._attention_backend_fa3_fp8_fallback is original
    assert consumer._attention_backend_fa3_fp8_fallback is original
    assert overrides.POST_PROCESS_PASSES[0] is original
    assert registry.get("flash_attention") is None
