# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU host-boundary contracts for combined built-in compatibility owners.

Run with HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES='' and this checkout on
PYTHONPATH: python -m pytest tests/adapter/test_vllm_builtin_health.py.
Real installed public selectors/spawn/fork are verified by separate probes.
"""
from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm as host
from hcu_adapter.bridges import (
    vllm_mhc, vllm_sparse_indexer, vllm_flashmla, vllm_flashmla_decode, vllm_pcp,
)


@pytest.mark.parametrize('boundary', ['runtime', 'forward'])
@pytest.mark.parametrize('failed_owner', [None, 'mhc', 'indexer', 'prefill', 'decode', 'pcp'])
def test_runtime_health_checks_each_builtin_owner(monkeypatch, failed_owner, boundary):
    calls = []

    def check(name):
        calls.append(name)
        if name == failed_owner:
            raise RuntimeError(name + ' owner unhealthy')

    monkeypatch.setattr(host, '_RUNTIME_ACTIONS',
                        SimpleNamespace(require_runtime_healthy=lambda: check('host'),
                                        runtime_action_in_progress=lambda: False,
                                        require_requested_extensions_healthy=lambda: check('requested')))
    monkeypatch.setattr(vllm_mhc, 'require_healthy', lambda: check('mhc'))
    monkeypatch.setattr(vllm_sparse_indexer, 'require_healthy', lambda: check('indexer'))
    monkeypatch.setattr(vllm_flashmla, 'require_healthy', lambda: check('prefill'))
    monkeypatch.setattr(vllm_flashmla_decode, 'require_healthy', lambda: check('decode'))
    monkeypatch.setattr(vllm_pcp, 'require_healthy', lambda: check('pcp'))
    guard = host.require_forward_healthy if boundary == 'forward' else host.require_runtime_healthy
    if failed_owner is None:
        guard()
    else:
        with pytest.raises(RuntimeError, match=failed_owner + ' owner unhealthy'):
            guard()
    owners = ['host', 'mhc', 'indexer', 'prefill', 'decode', 'pcp']
    expected = owners if failed_owner is None else owners[:owners.index(failed_owner) + 1]
    if boundary == 'forward' and failed_owner is None:
        expected.append('requested')
    assert calls == expected


def test_runtime_defers_owner_checks_until_activation_callback_returns(monkeypatch):
    actions = host.ProcessLocalActions()
    monkeypatch.setattr(host, '_RUNTIME_ACTIONS', actions)
    checks = []
    for owner in (vllm_mhc, vllm_sparse_indexer, vllm_flashmla, vllm_flashmla_decode, vllm_pcp):
        monkeypatch.setattr(owner, 'require_healthy', lambda: checks.append('owner'))

    def activate():
        host.require_runtime_healthy()
        assert checks == []

    actions.run_once('vllm-runtime', activate)
    host.require_runtime_healthy()
    assert checks == ['owner'] * 5
