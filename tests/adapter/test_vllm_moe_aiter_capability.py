# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Vendor query boundary without importing AITER or pretending to have a GPU."""

from types import SimpleNamespace

import pytest

from hcu_adapter.bridges import vllm_moe_aiter as bridge


@pytest.fixture(autouse=True)
def clear_query_cache():
    bridge._raw_layout_support.cache_clear()
    yield
    bridge._raw_layout_support.cache_clear()


def install_router(monkeypatch, query):
    router = SimpleNamespace(
        FusedMoeProblem=lambda **values: SimpleNamespace(**values),
        feasible_layouts=query,
    )

    def load(name):
        assert name == "aiter.fused_moe_router"
        return router

    monkeypatch.setattr(bridge, "import_module", load)
    return router


def arguments(**changes):
    values = dict(pid=101, architecture="gfx938", dtype="bf16", experts=256,
                  hidden=2048, intermediate=512, topk=8, activation="silu")
    values.update(changes)
    return values


@pytest.mark.parametrize("architecture", ("gfx928", "gfx92a", "gfx936", "gfx938"))
@pytest.mark.parametrize("dtype,activation", (("bf16", "silu"), ("fp16", "gelu")))
def test_query_carries_real_identity_and_canonical_raw_problem(
    monkeypatch, architecture, dtype, activation
):
    observed = []

    def query(problem):
        observed.append(vars(problem))
        return frozenset({"raw", "asm_shuffle"})

    install_router(monkeypatch, query)
    assert bridge._raw_layout_support(**arguments(
        architecture=architecture, dtype=dtype, activation=activation
    )) == (True, None)
    assert observed == [dict(
        arch=architecture, M=1, topk=8, E=256, model_dim=2048, inter_dim=512,
        dtype=dtype, q_dtype_a="none", q_dtype_w="none", quant_kind="none",
        quant_type="no", g1u1=True, activation=activation,
        weight_layout="raw", local_E=256,
    )]


@pytest.mark.parametrize("layouts", (frozenset(), frozenset({"asm_shuffle", "marlin"})))
def test_absent_raw_route_is_refused(monkeypatch, layouts):
    install_router(monkeypatch, lambda problem: layouts)
    supported, reason = bridge._raw_layout_support(**arguments())
    assert not supported
    assert "no raw-weight route" in reason and "gfx938" in reason


def test_explicit_vendor_refusal_is_returned(monkeypatch):
    def query(problem):
        raise NotImplementedError("no matching native configuration")

    install_router(monkeypatch, query)
    supported, reason = bridge._raw_layout_support(**arguments())
    assert not supported and "no matching native configuration" in reason


@pytest.mark.parametrize("error", (RuntimeError, AttributeError, ValueError))
def test_unexpected_vendor_query_errors_propagate(monkeypatch, error):
    def query(problem):
        raise error("broken vendor query")

    install_router(monkeypatch, query)
    with pytest.raises(error, match="broken vendor query"):
        bridge._raw_layout_support(**arguments())


def test_missing_vendor_is_refused(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named aiter")

    monkeypatch.setattr(bridge, "import_module", missing)
    supported, reason = bridge._raw_layout_support(**arguments())
    assert not supported and "capability API unavailable" in reason


@pytest.mark.parametrize("missing", ("FusedMoeProblem", "feasible_layouts"))
def test_older_vendor_without_query_api_is_refused(monkeypatch, missing):
    router = install_router(monkeypatch, lambda problem: frozenset({"raw"}))
    delattr(router, missing)
    supported, reason = bridge._raw_layout_support(**arguments())
    assert not supported and "capability API unavailable" in reason


@pytest.mark.parametrize("change", (
    {"pid": 102}, {"architecture": "gfx928"}, {"dtype": "fp16"},
    {"experts": 8}, {"hidden": 256}, {"intermediate": 128},
    {"topk": 2}, {"activation": "gelu"},
))
def test_cache_is_process_and_configuration_specific(monkeypatch, change):
    calls = []

    def query(problem):
        calls.append(problem)
        return frozenset({"raw"})

    install_router(monkeypatch, query)
    assert bridge._raw_layout_support(**arguments()) == (True, None)
    assert bridge._raw_layout_support(**arguments()) == (True, None)
    assert len(calls) == 1
    assert bridge._raw_layout_support(**arguments(**change)) == (True, None)
    assert len(calls) == 2
