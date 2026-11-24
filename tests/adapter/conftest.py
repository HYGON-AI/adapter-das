# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host scoping for the vLLM-side adapter tests.

The vLLM-host suite (``test_vllm_*.py``, ``cases/vllm_*.py`` and the
vLLM-Worker ``test_model_mutation.py``) asserts process-level isolation
-- its ``isolated`` fixtures demand a CUDA-clean interpreter and the
bridges installed for host ``vllm``. Inside the sglang L1 kernels segment
one pytest process hosts sglang and has initialized CUDA on the device
tests, so on images that ship BOTH frameworks (2026-11-16 sglang:0.5.22
image carries vllm 0.33) these tests cannot run faithfully: they errored
en masse at setup (``assert not torch.cuda.is_initialized()``) instead of
being skipped as they were on vllm-less images. They still run -- and
must stay green -- in the vllm lane, where the host is pinned to vllm
and each file gets a fresh interpreter.

An explicitly device-hidden HOSTnone process also runs these CPU contracts.
The L1 cpu_vllm segment and documented standalone CPU commands use that
boundary; skipping them would silently remove the transferred coverage.

Also batches the ``_cpu_process`` clean-subprocess reruns: files whose
tests each spawn their own interpreter when the session loaded a host
(69 ct_w4a8 tests alone cost ~540 s of interpreter+torch startup).
One subprocess for the whole batch restores the same isolation for the
price of a single startup.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# vLLM-Worker machinery without the test_vllm_ prefix.
_EXTRA_VLLM_FILES = {"test_model_mutation.py"}

# CPU oracle suites that rerun themselves via _cpu_process when a host
# is loaded in the session process.
_CPU_RERUN_FILES = {
    "test_checkpoint_quantization_reference.py",
    "test_ct_w4a8_native_reference.py",
    "test_ct_w4a8_reference.py",
    "test_l15_checkpoint_dtype_probe.py",
    "test_l15_checkpoint_reference.py",
}


@pytest.fixture
def cpu_builtin_owner_health(monkeypatch):
    """Keep CPU selector/Worker doubles independent of native owner startup.

    Leave the runtime latch and required-extension health checks real. Built-in
    owner aggregation and drift are exercised by their dedicated suites.
    """
    from importlib import import_module

    for name in ("vllm_mhc", "vllm_sparse_indexer", "vllm_flashmla",
                 "vllm_flashmla_decode", "vllm_pcp"):
        owner = import_module("hcu_adapter.bridges." + name)
        monkeypatch.setattr(owner, "require_healthy", lambda: None)


def _is_vllm_host_file(path: str) -> bool:
    name = os.path.basename(path)
    if name in _EXTRA_VLLM_FILES:
        return True
    if name.startswith("test_vllm_"):
        return True
    parent = os.path.basename(os.path.dirname(path))
    return parent == "cases" and name.startswith("vllm_")


def _host_loaded() -> bool:
    return any(
        name == "sglang" or name.startswith(("sglang.", "hcu_adapter.bridges."))
        for name in sys.modules
    )


def _hidden_cpu_process() -> bool:
    if os.environ.get("HCU_ADAPTER_HOST") != "none":
        return False
    if os.environ.get("_HCU_ADAPTER_ACTIVE_HOST") not in (None, "none"):
        return False
    if any(name == "sglang" or name.startswith("sglang.") for name in sys.modules):
        return False
    if not all(os.environ.get(key) in ("", "-1") for key in
               ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        return False
    torch = sys.modules.get("torch")
    return torch is None or not torch.cuda.is_initialized()


def _batch_clean_cpu(items) -> None:
    """Run every _cpu_process suite in ONE clean interpreter, then skip
    the in-process copies. On failure the items are left alone: their own
    per-test clean rerun then reproduces the real failure output."""
    nodeids = sorted({item.nodeid for item in items})
    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ,
        HCU_ADAPTER_HOST="none", VLLM_PLUGINS="none",
        HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        PYTHONPATH=str(root), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *nodeids],
        cwd=root, env=environment, capture_output=True, text=True, timeout=3600,
    )
    tail = (result.stdout + result.stderr).strip().splitlines()
    summary = tail[-1] if tail else f"rc={result.returncode}"
    if result.returncode != 0:
        print(
            f"[cpu-batch] clean interpreter FAILED rc={result.returncode} "
            f"({len(nodeids)} tests): {summary}",
            file=sys.stderr,
        )
        return
    skip = pytest.mark.skip(
        reason=f"executed in one clean CPU interpreter ({len(nodeids)} tests): {summary}"
    )
    for item in items:
        item.add_marker(skip)


def pytest_collection_modifyitems(config, items):
    active = os.environ.get("_HCU_ADAPTER_ACTIVE_HOST")
    if active == "vllm" or _hidden_cpu_process():
        return
    reason = (
        "vllm-host suite: needs a CUDA-clean process with the vllm host "
        f"bridges (active host here: {active or 'none'})"
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if _is_vllm_host_file(str(item.path)):
            item.add_marker(skip)
    if _host_loaded():
        batch = [
            item for item in items
            if os.path.basename(str(item.path)) in _CPU_RERUN_FILES
        ]
        if batch:
            _batch_clean_cpu(batch)
