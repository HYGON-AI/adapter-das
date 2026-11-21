# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Device admission for Core's existing communication-CU configuration."""

import logging
import os
from contextlib import contextmanager

from hcu_adapter.infra import hcu_config, hcu_env, registry

LOG = logging.getLogger("hcu.adapter.vllm.dbo")
_ENV = "VLLM_DBO_COMM_SMS"


@contextmanager
def communication_cu_config(parallel_config, total_cus):
    """Publish an HCU default only while Core constructs its SM controller.

    ``total_cus`` is the CU count of the already-bound worker device. The
    caller is HCU-only; this is not a replacement for Core's SM controller.
    Core retains range validation and backend-specific limits/overrides.
    """
    from vllm import envs

    if not parallel_config.use_ubatching:
        yield
        return
    if (parallel_config.enable_dbo
            and parallel_config.all2all_backend == "deepep_high_throughput"):
        # Core's ROCm branch already forces zero, even for explicit inputs.
        yield
        return
    if (parallel_config.enable_expert_parallel
            and parallel_config.all2all_backend != "allgather_reducescatter"):
        # Other managers can cap max_sms_used() inside Core. Do not override
        # a valid backend-specific budget before Core has applied that cap.
        yield
        return
    value, source = hcu_env.resolve(_ENV, host="vllm")
    if envs.is_set(_ENV):
        # Including empty/malformed inputs: let the official parser reject.
        LOG.info("HCU DBO: %s=%s (explicit); preserving official input",
                 _ENV, value if source == "explicit" else hcu_env.environment_value(_ENV))
        yield
        return
    if hcu_config.load()["ops"].get("vllm_dbo_cu", "auto") == "stock":
        yield
        return
    official_default = envs.VLLM_DBO_COMM_SMS
    if total_cus <= 0:
        raise ValueError("DBO requires the bound device's positive CU count")
    if official_default < total_cus:
        yield
        return
    # Zero is Core's supported unpartitioned path, not a guessed budget.
    selected = 0

    try:
        os.environ[_ENV] = str(selected)
        if envs.VLLM_DBO_COMM_SMS != selected:
            raise RuntimeError(
                "HCU DBO communication CU policy must run before Core env "
                "caching; use a fresh worker process"
            )
        registry.register("vllm_dbo_cu", "auto", "vllm.envs.VLLM_DBO_COMM_SMS",
                          registry.MECH_SEAM)
        LOG.info("HCU DBO: %s=%s (auto; official default=%s), device CUs=%s",
                 _ENV, selected, official_default, total_cus)
        yield
    finally:
        # No sticky auto value across later models or failed model loads.
        os.environ.pop(_ENV, None)


@contextmanager
def worker_communication_cu_config(worker):
    """Query only the device already selected by official Worker.init_device."""
    parallel_config = worker.vllm_config.parallel_config
    if not getattr(parallel_config, "use_ubatching", False):
        yield
        return
    if worker.use_v2_model_runner:
        # anchor-baseline(③): OpenDAS v0.33/c440c08 freezes worker envs
        # after load_model; V2 creates SM control later in KV initialization.
        # Leave that path official-owned; an explicit input remains supported.
        yield
        return
    from vllm.platforms import current_platform
    from vllm.utils.platform_utils import num_compute_units

    if not current_platform.is_rocm():
        yield
        return
    with communication_cu_config(parallel_config, num_compute_units(worker.device.index)):
        yield
