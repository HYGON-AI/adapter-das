# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Official config/env contracts; device constructor tests live separately."""

import os

import pytest

pytest.importorskip("vllm")
from vllm import envs
from vllm.config import ParallelConfig

from hcu_adapter.bridges.vllm_dbo import communication_cu_config
from hcu_adapter.infra import hcu_config

NAME = "VLLM_DBO_COMM_SMS"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    envs.disable_envs_cache()
    # Use actual config discovery rather than injecting private config state.
    path = tmp_path / "config.yaml"
    path.write_text("version: 1\n")
    with monkeypatch.context() as setup:
        setup.delenv(NAME, raising=False)
        setup.setenv("HCU_CONFIG", str(path))
        hcu_config.load(force=True)
        yield path
    envs.disable_envs_cache()
    hcu_config.load(force=True)


def config(**kwargs):
    return ParallelConfig(enable_dbo=True, **kwargs)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_official_default_relative_to_actual_device_budget(delta):
    default = envs.VLLM_DBO_COMM_SMS
    budget = default + delta
    with communication_cu_config(config(), budget):
        assert envs.VLLM_DBO_COMM_SMS == (default if delta > 0 else 0)
        assert envs.is_set(NAME) == (delta <= 0)
    assert NAME not in os.environ
    assert envs.VLLM_DBO_COMM_SMS == default


@pytest.mark.parametrize("value", ["0", "1", "64", "999", "-1", "", "bad"])
def test_explicit_input_remains_official_owned(monkeypatch, value):
    monkeypatch.setenv(NAME, value)
    with communication_cu_config(config(), 64):
        assert os.environ[NAME] == value
        if value in ("", "bad"):
            with pytest.raises(ValueError):
                _ = envs.VLLM_DBO_COMM_SMS
        else:
            assert envs.VLLM_DBO_COMM_SMS == int(value)
    assert os.environ[NAME] == value


@pytest.mark.parametrize("enable_dbo,ubatch_size,expected", [
    (False, 1, False), (False, 2, True), (True, 1, True),
])
def test_both_official_microbatch_modes(enable_dbo, ubatch_size, expected):
    parallel = ParallelConfig(enable_dbo=enable_dbo, ubatch_size=ubatch_size)
    with communication_cu_config(parallel, envs.VLLM_DBO_COMM_SMS):
        assert envs.is_set(NAME) == expected
    assert NAME not in os.environ


@pytest.mark.parametrize("explicit", [None, "", "bad", "999"])
def test_core_deepep_ht_override_is_not_preempted(monkeypatch, explicit):
    if explicit is not None:
        monkeypatch.setenv(NAME, explicit)
    before = dict(os.environ)
    with communication_cu_config(config(all2all_backend="deepep_high_throughput"), 64):
        assert dict(os.environ) == before
    assert dict(os.environ) == before


@pytest.mark.parametrize("backend", ["deepep_low_latency", "deepep_high_throughput", "deepep_v2"])
def test_other_ep_backend_caps_remain_core_owned(backend):
    parallel = config(enable_expert_parallel=True, all2all_backend=backend)
    with communication_cu_config(parallel, envs.VLLM_DBO_COMM_SMS):
        assert NAME not in os.environ


def test_ag_rs_ep_admission():
    parallel = config(enable_expert_parallel=True, all2all_backend="allgather_reducescatter")
    with communication_cu_config(parallel, envs.VLLM_DBO_COMM_SMS):
        assert envs.VLLM_DBO_COMM_SMS == 0


@pytest.mark.parametrize("backend", ["naive", "pplx"])
def test_removed_alias_uses_official_normalized_backend(backend):
    parallel = config(enable_expert_parallel=True, all2all_backend=backend)
    assert parallel.all2all_backend == "allgather_reducescatter"
    with communication_cu_config(parallel, envs.VLLM_DBO_COMM_SMS):
        assert envs.VLLM_DBO_COMM_SMS == 0


@pytest.mark.parametrize("policy", ["auto", "stock"])
def test_yaml_selects_admission_policy(isolated, policy):
    isolated.write_text(f"version: 1\nops:\n  vllm_dbo_cu: {policy}\n")
    hcu_config.load(force=True)
    with communication_cu_config(config(), 64):
        assert envs.is_set(NAME) == (policy == "auto")
    assert NAME not in os.environ


def test_explicit_over_yaml(monkeypatch, isolated):
    isolated.write_text("version: 1\nops:\n  vllm_dbo_cu: auto\n")
    hcu_config.load(force=True)
    monkeypatch.setenv(NAME, "7")
    with communication_cu_config(config(), 64):
        assert envs.VLLM_DBO_COMM_SMS == 7
    assert os.environ[NAME] == "7"


def test_exception_nested_and_sequential_model_restoration():
    default = envs.VLLM_DBO_COMM_SMS
    parallel = config()
    with pytest.raises(KeyboardInterrupt):
        with communication_cu_config(parallel, default):
            with communication_cu_config(parallel, default + 1):
                assert envs.VLLM_DBO_COMM_SMS == 0
            assert envs.VLLM_DBO_COMM_SMS == 0
            raise KeyboardInterrupt
    assert NAME not in os.environ
    with communication_cu_config(parallel, default + 1):
        assert envs.VLLM_DBO_COMM_SMS == default
        assert NAME not in os.environ


def test_cached_old_default_refuses_without_clearing_cache():
    parallel = config()
    default = envs.VLLM_DBO_COMM_SMS
    envs.enable_envs_cache()
    with pytest.raises(RuntimeError, match="before Core env caching"):
        with communication_cu_config(parallel, default):
            pytest.fail("entered Core with a stale budget")
    assert envs._is_envs_cache_enabled()
    assert envs.VLLM_DBO_COMM_SMS == default
    assert NAME not in os.environ


def test_cached_explicit_value_is_preserved(monkeypatch):
    monkeypatch.setenv(NAME, "0")
    parallel = config()
    envs.enable_envs_cache()
    with communication_cu_config(parallel, 64):
        assert envs.VLLM_DBO_COMM_SMS == 0
    assert envs._is_envs_cache_enabled()
    assert os.environ[NAME] == "0"
