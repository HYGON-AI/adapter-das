# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""CPU execution-boundary contracts using the real official Worker/Graph manager.

Run with the target vLLM installed and GPU visibility disabled. Only the GPU
graph and communicator are doubles; no checkpoint or device acceptance implied.
"""
from contextlib import nullcontext
import inspect
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

import pytest

pytest.importorskip("vllm")
# Only the official ROCm import-time architecture descriptor is synthetic.
# Restore device queries before exercising the real Worker/Graph APIs.
with patch.object(torch.cuda, "get_device_properties", return_value=NS(gcnArchName="gfx936")):
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker import gpu_worker
    from vllm.v1.worker.gpu import cudagraph_utils as cg

from hcu_adapter.infra import extensions as ext
from hcu_adapter.bridges import vllm as bridge
from hcu_adapter.bridges.vllm_worker import Worker


@pytest.fixture
def execution(monkeypatch, cpu_builtin_owner_health):
    actions = bridge.ProcessLocalActions()
    monkeypatch.setattr(bridge, "_RUNTIME_ACTIONS", actions)
    for name in ("_lazy_init", "init", "current_device", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name,
                            lambda *a, **k: pytest.fail("CPU contract queried GPU"))
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: NS(is_first_rank=True))
    offload = Mock()
    monkeypatch.setattr(cg, "get_offloader", lambda: offload)
    replay = Mock()
    desc = cg.BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, 1)
    manager = object.__new__(cg.CudaGraphManager)
    manager.graphs = {desc: NS(replay=replay)}

    def run(scheduler_output, intermediate_tensors):
        assert intermediate_tensors is None
        manager.run_fullgraph(desc)
        return None

    worker = object.__new__(Worker)
    pending_send = Mock()
    worker._pp_send_work = [pending_send]
    worker.annotate_profile = lambda _: nullcontext()
    worker.model_runner = NS(execute_model=run, is_pooling_model=False)
    worker.use_v2_model_runner = True
    return NS(actions=actions, worker=worker, replay=replay, offload=offload,
              pending_send=pending_send, schedule=NS(total_num_scheduled_tokens=1))


def enroll(actions, checker=None, *, requested=True):
    context = ext.ExtensionContext("cpu", None, (), {"worker-test"} if requested else ())
    points = [NS(name="worker-owner", load=lambda: lambda: ext.ExtensionSpec(
        "declared-worker-owner", 1, 0, {"worker-test"}, lambda _: None, checker))]

    def boot():
        reports = ext.activate_extensions(context, points,
                                         health_check_sink=actions.record_extension_health)
        actions.record_extension_reports(reports)
        actions.record_forward_extension_policy(context)

    actions.run_once(actions._runtime_action, boot)


def test_worker_preserves_official_execute_signature():
    assert inspect.signature(Worker.execute_model) == inspect.signature(gpu_worker.Worker.execute_model)


@pytest.mark.parametrize("requested", [False, True])
def test_healthy_worker_delegates_to_official_fullgraph(execution, requested):
    checker = Mock(return_value=None)
    enroll(execution.actions, checker, requested=requested)
    assert execution.worker.execute_model(execution.schedule) is None
    execution.pending_send.wait.assert_called_once_with()
    execution.offload.sync_prev_onload.assert_called_once_with()
    execution.replay.assert_called_once_with()
    assert checker.call_count == int(requested)


@pytest.mark.parametrize("fault", ["checker", "report", "unactivated", "unenrolled"])
def test_required_owner_failure_stops_before_send_offload_and_replay(execution, fault):
    broken = [False]

    def checker():
        if broken[0]:
            raise RuntimeError("required owner drift")

    if fault in ("checker", "report"):
        enroll(execution.actions, checker)
        if fault == "checker":
            broken[0] = True
        else:
            # Same-object republication is a generation change, too.
            execution.actions.record_extension_reports(execution.actions.extension_reports())
    elif fault == "unenrolled":
        execution.actions.run_once(execution.actions._runtime_action, lambda: None)
    for _ in range(2):
        with pytest.raises(bridge.LatchedVllmRuntimeError):
            execution.worker.execute_model(execution.schedule)
        # Restoring the provider cannot repair a process after refusal.
        broken[0] = False
    execution.pending_send.wait.assert_not_called()
    execution.offload.sync_prev_onload.assert_not_called()
    execution.replay.assert_not_called()


def test_public_platform_guard_is_absent_from_official_fullgraph_replay(execution, monkeypatch):
    """Lock in the actual upstream path that needs the outer Worker boundary."""
    guard = Mock(side_effect=AssertionError("platform forward hook entered"))
    monkeypatch.setattr(bridge, "require_forward_healthy", guard)
    # The official manager's replay works without calling the platform hook.
    # This control explains why platform-only CPU tests miss FULL replay.
    cg.CudaGraphManager.run_fullgraph(
        NS(graphs={cg.BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, 1):
                   NS(replay=execution.replay)}),
        cg.BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, 1))
    guard.assert_not_called()
    execution.replay.assert_called_once_with()


@pytest.mark.parametrize("drift", [False, True])
def test_factory_selected_runner_retains_required_guard_before_fullgraph(execution, monkeypatch, drift):
    """The two Worker overrides compose through the real factory and replay APIs."""
    import importlib
    from hcu_adapter.bridges import vllm_model_runner as factories
    importlib.reload(factories)
    try:
        configuration = object()
        execution.worker.vllm_config = configuration
        execution.worker.device = "cpu-fixture"
        runner = execution.worker.model_runner
        registration = factories.register_model_runner_factory(
            "worker-test", matches=lambda cfg: cfg is configuration,
            create=lambda cfg, device: runner if device == "cpu-fixture" else None)
        enroll(execution.actions, lambda: factories.require_model_runner_factory(registration))
        assert execution.worker._make_model_runner() is runner
        if drift:
            monkeypatch.setattr(factories, "_factories", {})
            with pytest.raises(bridge.LatchedVllmRuntimeError):
                execution.worker.execute_model(execution.schedule)
            execution.pending_send.wait.assert_not_called()
            execution.offload.sync_prev_onload.assert_not_called()
            execution.replay.assert_not_called()
        else:
            assert execution.worker.execute_model(execution.schedule) is None
            execution.pending_send.wait.assert_called_once_with()
            execution.offload.sync_prev_onload.assert_called_once_with()
            execution.replay.assert_called_once_with()
    finally:
        importlib.reload(factories)
