# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU binding contracts for both supported upstream INT8 call sites.

Run with HCU_ADAPTER_HOST=none; vendor/custom-op doubles execute no device math.
"""
from types import ModuleType, SimpleNamespace
import os
import select
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from hcu_adapter.infra import registry
from hcu_adapter.infra import hcu_config
from hcu_adapter.bridges import vllm as host, vllm_scaled_mm as bridge


@pytest.fixture
def startup(monkeypatch):
    module = ModuleType('vllm.model_executor.kernels.linear.scaled_mm.triton')
    exec('''
def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    return 'official'
class TritonFP8ScaledMMLinearKernel:
    def apply_scaled_mm(self, *args):
        return triton_scaled_mm(*args)
class TritonInt8ScaledMMLinearKernel:
    def apply_weights(self, *args):
        return triton_scaled_mm(*args)
''', module.__dict__)
    def vendor(a, b, scale_a, scale_b, m, n, k, transpose_flag, out_dtype, bias):
        raise AssertionError('CPU tests must not execute vendor math')
    monkeypatch.setattr(bridge, 'import_module', lambda name: {
        module.__name__: module,
        'lightop.gemm_ops': SimpleNamespace(hipblaslt_w8a8_channelwise_gemm=vendor),
    }[name])
    monkeypatch.setattr(host, 'detect_hcu', lambda: host.HcuDetection(True, 'gfx938', 'fixture'))
    monkeypatch.setattr(hcu_config, '_STATE', {'loaded': True, 'path': None,
                         'ops': {}, 'overrides': {}, 'hosts': {}})
    monkeypatch.delenv('HCU_DISABLE_OPS', raising=False)
    monkeypatch.setattr(registry, '_RECORDS', {})
    monkeypatch.setattr(registry, '_BOUND', {})
    publications = []
    class Op:
        def register_fake(self, fake):
            pass
        def __call__(self, *args):
            return 'custom-op'
    def custom_op(*args, **kwargs):
        publications.append(args[0])
        return Op()
    monkeypatch.setattr(torch.library, 'custom_op', custom_op)
    return module, publications


@pytest.mark.parametrize('site', ['v032-direct', 'v033-custom-op'])
@pytest.mark.parametrize('choice', ['auto', 'lightop'])
def test_both_supported_int8_sites_keep_official_methods(startup, site, choice):
    module, publications = startup
    hcu_config._STATE['ops']['vllm_scaled_mm'] = choice
    if site == 'v033-custom-op':
        exec('''
def _int8_scaled_mm_func(*args):
    return triton_scaled_mm(*args)
def apply_weights(self, *args):
    return _int8_scaled_mm_func(*args)
TritonInt8ScaledMMLinearKernel.apply_weights = apply_weights
''', module.__dict__)
    int8_method = module.TritonInt8ScaledMMLinearKernel.apply_weights
    fp8_method = module.TritonFP8ScaledMMLinearKernel.apply_scaled_mm
    assert bridge.install()
    assert module.TritonInt8ScaledMMLinearKernel.apply_weights is int8_method
    assert module.TritonFP8ScaledMMLinearKernel.apply_scaled_mm is fp8_method
    assert module.TritonInt8ScaledMMLinearKernel().apply_weights(*range(5)) == 'custom-op'
    assert module.TritonFP8ScaledMMLinearKernel().apply_scaled_mm(*range(5)) == 'custom-op'
    assert bridge.install() is False
    assert len(publications) == 1


@pytest.mark.parametrize('choice', ['auto', 'lightop'])
def test_unknown_host_site_stays_stock_in_auto_and_refuses_explicit(startup, choice):
    module, publications = startup
    hcu_config._STATE['ops']['vllm_scaled_mm'] = choice
    del module.TritonInt8ScaledMMLinearKernel
    original = module.triton_scaled_mm
    if choice == 'auto':
        assert bridge.install() is False
    else:
        with pytest.raises(host.VllmCompatibilityError, match='call site'):
            bridge.install()
    assert module.triton_scaled_mm is original
    assert registry.get('vllm_scaled_mm') is None
    assert not publications


@pytest.fixture
def executing_startup(startup, monkeypatch):
    """Run the actual routed dispatch, doubling only device computation."""
    from hcu_adapter.kernels import scaled_mm_lightop as kernel

    module, publications = startup
    imports = []
    base_import = bridge.import_module
    state = {'eligible': True, 'provider': None, 'error': None}

    def importing(name):
        if name == 'lightop.gemm_ops':
            imports.append(name)
            if state['error'] is not None:
                raise state['error']
            if state['provider'] is not None:
                return SimpleNamespace(hipblaslt_w8a8_channelwise_gemm=state['provider'])
        return base_import(name)

    class Op:
        def __init__(self, dispatch):
            self.dispatch = dispatch

        def register_fake(self, fake):
            state['fake'] = fake

        def __call__(self, *args):
            return self.dispatch(*args)

    def custom_op(name, dispatch, **kwargs):
        publications.append(name)
        return Op(dispatch)

    monkeypatch.setattr(bridge, 'import_module', importing)
    monkeypatch.setattr(torch.library, 'custom_op', custom_op)
    monkeypatch.setattr(kernel, 'supported', lambda *args: state['eligible'])
    monkeypatch.setattr(kernel, 'scaled_mm', lambda *args, gemm: ('vendor', gemm))
    return module, state, imports


def test_auto_does_not_import_vendor_until_eligible_execution(executing_startup):
    module, state, imports = executing_startup
    assert bridge.install()
    assert not imports
    state['eligible'] = False
    assert module.triton_scaled_mm(*range(5)) == 'official'
    assert not imports
    state['eligible'] = True
    first = module.triton_scaled_mm(*range(5))
    assert first[0] == 'vendor' and callable(first[1])
    assert module.triton_scaled_mm(*range(5)) == first
    assert len(imports) == 1


@pytest.mark.parametrize('error', [ImportError('missing wheel'), AttributeError('missing entry')])
def test_auto_missing_vendor_keeps_official_execution(executing_startup, error):
    module, state, imports = executing_startup
    state['error'] = error
    assert bridge.install()
    assert not imports
    for _ in range(3):
        assert module.triton_scaled_mm(*range(5)) == 'official'
    assert len(imports) == 1


@pytest.mark.parametrize('error', [ImportError('missing wheel'), AttributeError('missing entry')])
def test_explicit_vendor_is_still_required_at_startup(executing_startup, error):
    module, state, imports = executing_startup
    hcu_config._STATE['ops']['vllm_scaled_mm'] = 'lightop'
    state['error'] = error
    original = module.triton_scaled_mm
    with pytest.raises(host.VllmCompatibilityError, match='explicitly requested'):
        bridge.install()
    assert len(imports) == 1
    assert module.triton_scaled_mm is original


def test_explicit_vendor_is_eager_but_imported_only_once(executing_startup):
    module, state, imports = executing_startup
    hcu_config._STATE['ops']['vllm_scaled_mm'] = 'lightop'
    assert bridge.install()
    assert len(imports) == 1
    assert module.triton_scaled_mm(*range(5))[0] == 'vendor'
    assert len(imports) == 1


def test_bad_auto_abi_latches_refusal_for_every_later_call(executing_startup):
    module, state, imports = executing_startup
    state['provider'] = lambda: None
    assert bridge.install()
    with pytest.raises(host.VllmCompatibilityError, match='ABI changed'):
        module.triton_scaled_mm(*range(5))
    state.update(provider=None, eligible=False)
    with pytest.raises(host.VllmCompatibilityError, match='ABI changed'):
        module.triton_scaled_mm(*range(5))
    with pytest.raises(host.VllmCompatibilityError, match='ABI changed'):
        module.triton_scaled_mm(*range(5), extra=True)
    assert len(imports) == 1


def test_concurrent_first_calls_resolve_one_provider(executing_startup, monkeypatch):
    module, state, imports = executing_startup
    original_import = bridge.import_module
    entered, release = Event(), Event()

    def blocking(name):
        if name == 'lightop.gemm_ops':
            entered.set()
            assert release.wait(5)
        return original_import(name)

    monkeypatch.setattr(bridge, 'import_module', blocking)
    assert bridge.install()
    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(module.triton_scaled_mm, *range(5))
        try:
            assert entered.wait(5)
            rest = [pool.submit(module.triton_scaled_mm, *range(5)) for _ in range(7)]
        finally:
            release.set()
        rows = [first.result(timeout=5)] + [f.result(timeout=5) for f in rest]
    assert all(row == rows[0] for row in rows) and rows[0][0] == 'vendor'
    assert len(imports) == 1


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires POSIX fork')
def test_child_refuses_inherited_inflight_provider_without_deadlock(executing_startup, monkeypatch):
    module, state, imports = executing_startup
    original_import = bridge.import_module
    entered, release = Event(), Event()

    def blocking(name):
        if name == 'lightop.gemm_ops':
            entered.set()
            assert release.wait(10)
        return original_import(name)

    monkeypatch.setattr(bridge, 'import_module', blocking)
    assert bridge.install()
    readfd, writefd = os.pipe()
    child = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(module.triton_scaled_mm, *range(5))
            try:
                assert entered.wait(5)
                child = os.fork()
                if child == 0:
                    os.close(readfd)
                    try:
                        module.triton_scaled_mm(*range(5))
                    except host.VllmCompatibilityError as error:
                        os.write(writefd, str(error).encode())
                    finally:
                        os._exit(0)
                os.close(writefd)
                assert select.select([readfd], [], [], 5)[0], 'child blocked on inherited lock'
                assert b'inherited incomplete vllm-scaled-mm-provider' in os.read(readfd, 4096)
            finally:
                release.set()
            assert future.result(timeout=5)[0] == 'vendor'
        assert os.waitpid(child, 0)[1] == 0
        child = None
    finally:
        os.close(readfd)
        if child is not None:
            os.kill(child, 9)
            os.waitpid(child, 0)
