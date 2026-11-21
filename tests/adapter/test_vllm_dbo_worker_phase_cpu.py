# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real thin Worker KV boundary after Core freezes its env."""

import pytest
import torch

from test_vllm_post_model_load import config, isolation  # noqa: F401
from hcu_adapter.bridges.vllm_worker import Worker, OfficialWorker
from hcu_adapter.bridges.vllm_dbo import worker_communication_cu_config
from vllm import envs
from vllm.utils import platform_utils


@pytest.mark.usefixtures("isolation")
def test_v1_kv_initialization_preserves_already_cached_environment(monkeypatch):
    cfg = config()
    cfg.parallel_config.enable_dbo = True
    # CPU boundary fixture, not device admission or a constructed runner.
    worker = object.__new__(Worker)
    worker.vllm_config = cfg
    worker.use_v2_model_runner = False
    worker.device = torch.device("cuda:0")
    monkeypatch.delenv("VLLM_DBO_COMM_SMS", raising=False)
    monkeypatch.setattr(platform_utils, "num_compute_units", lambda index: 64)
    default = envs.VLLM_DBO_COMM_SMS
    envs.enable_envs_cache()
    sentinel = object()
    called = []

    def initialize(self, cache):
        assert self is worker and cache is sentinel
        assert envs._is_envs_cache_enabled()
        assert envs.VLLM_DBO_COMM_SMS == default
        called.append(cache)
        return sentinel

    monkeypatch.setattr(OfficialWorker, "initialize_from_config", initialize)
    assert worker.initialize_from_config(sentinel) is sentinel
    assert called == [sentinel]


@pytest.mark.usefixtures("isolation")
def test_v2_load_does_not_guess_a_later_cached_controller_budget(monkeypatch):
    cfg = config()
    cfg.parallel_config.enable_dbo = True
    worker = object.__new__(Worker)
    worker.vllm_config = cfg
    worker.use_v2_model_runner = True
    monkeypatch.delenv("VLLM_DBO_COMM_SMS", raising=False)
    default = envs.VLLM_DBO_COMM_SMS
    # Deliberately no device field: the skipped adaptation must not query HIP.
    with worker_communication_cu_config(worker):
        assert envs.VLLM_DBO_COMM_SMS == default
        assert not envs.is_set("VLLM_DBO_COMM_SMS")
