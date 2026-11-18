# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU host-boundary contracts for combined built-in compatibility owners.

Run with HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES='' and this checkout on
PYTHONPATH: python -m pytest tests/adapter/test_vllm_builtin_health.py.
Real installed public selectors/spawn/fork are verified by separate probes.
"""
from importlib import import_module

import pytest

from hcu_adapter.bridges import vllm as host

OWNERS = ('mhc', 'sparse_indexer', 'flashmla', 'flashmla_decode', 'pcp')


@pytest.mark.parametrize('failed_owner', [None, *OWNERS])
def test_runtime_health_checks_each_builtin_owner(monkeypatch, failed_owner):
    calls = []

    def check(name):
        calls.append(name)
        if name == failed_owner:
            raise RuntimeError(name + ' owner unhealthy')

    lifecycle = host.ProcessLocalActions()
    monkeypatch.setattr(lifecycle, 'require_runtime_healthy', lambda: check('host'))
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', lifecycle)
    for name in OWNERS:
        module = import_module('hcu_adapter.bridges.vllm_' + name)
        monkeypatch.setattr(module, 'require_healthy', lambda name=name: check(name))
    if failed_owner is None:
        host.require_runtime_healthy()
    else:
        with pytest.raises(RuntimeError, match=failed_owner + ' owner unhealthy'):
            host.require_runtime_healthy()
    checked = OWNERS if failed_owner is None else OWNERS[:OWNERS.index(failed_owner) + 1]
    expected = ['host', *checked]
    assert calls == expected


def test_owner_health_waits_for_real_runtime_transaction(monkeypatch):
    lifecycle = host.ProcessLocalActions()
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', lifecycle)
    calls = []

    def not_ready():
        calls.append('mhc')
        raise RuntimeError('requested owner has not activated')

    module = import_module('hcu_adapter.bridges.vllm_mhc')
    monkeypatch.setattr(module, 'require_healthy', not_ready)
    lifecycle.run_once('vllm-runtime', host.require_runtime_healthy)
    assert calls == []
    with pytest.raises(RuntimeError, match='has not activated'):
        host.require_runtime_healthy()
    assert calls == ['mhc']
