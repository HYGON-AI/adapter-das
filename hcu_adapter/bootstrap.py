# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Host selection for the automatic HCU adapter bootstrap."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import PurePath

HOST_ENV = "HCU_ADAPTER_HOST"
ACTIVE_HOST_ENV = "_HCU_ADAPTER_ACTIVE_HOST"

_HOSTS = frozenset({"sglang", "vllm", "deepinfer"})
_PUBLIC_CHOICES = _HOSTS | {"auto", "none"}
_PROPAGATED_CHOICES = _HOSTS | {"none"}
_installed_pid: int | None = None

LOG = logging.getLogger("hcu.adapter")


def _choice(raw: object, *, variable: str, allowed: frozenset[str]) -> str:
    value = str(raw or "").strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{variable} must be one of: {choices}; got {raw!r}")
    return value


def _host_from_argv(argv: Sequence[str]) -> str | None:
    if not argv:
        return None

    launcher = str(argv[0]).strip().lower()
    basename = PurePath(launcher).name
    if basename in _HOSTS:
        return basename
    if "/" not in launcher:
        module_root = launcher.split(".", 1)[0]
        if module_root in _HOSTS:
            return module_root

    for index, raw in enumerate(argv[:-1]):
        token = str(raw).strip().lower()
        if token == "-m":
            module_root = str(argv[index + 1]).split(".", 1)[0].lower()
            if module_root in _HOSTS:
                return module_root
    return None


def _host_from_loaded(loaded_modules: Iterable[str]) -> str | None:
    roots = {str(name).split(".", 1)[0] for name in loaded_modules}
    present = roots & _HOSTS
    if len(present) == 1:
        return next(iter(present))
    if {"sglang", "vllm"}.issubset(present):
        return "sglang"
    return None


def _detect_host(
    argv: Sequence[str],
    loaded_modules: Iterable[str],
    available: Iterable[str],
    environment: Mapping[str, str],
) -> tuple[str | None, str]:
    public = _choice(
        environment.get(HOST_ENV, "auto"),
        variable=HOST_ENV,
        allowed=_PUBLIC_CHOICES,
    )
    if public != "auto":
        return (None if public == "none" else public), "explicit"

    inherited_raw = environment.get(ACTIVE_HOST_ENV)
    if inherited_raw:
        inherited = _choice(
            inherited_raw,
            variable=ACTIVE_HOST_ENV,
            allowed=_PROPAGATED_CHOICES,
        )
        return (None if inherited == "none" else inherited), "inherited"

    intended = _host_from_argv(argv)
    if intended is not None:
        return intended, "argv"

    loaded = _host_from_loaded(loaded_modules)
    if loaded is not None:
        return loaded, "loaded-module"

    present = set(available) & _HOSTS
    if len(present) == 1:
        return next(iter(present)), "availability"
    if {"sglang", "vllm"}.issubset(present):
        return "sglang", "legacy-default"
    return None, "unavailable"


def detect_host(
    argv: Sequence[str],
    loaded_modules: Iterable[str],
    available: Iterable[str],
    environment: Mapping[str, str],
) -> str | None:
    """Select exactly one framework without importing either framework."""
    host, _ = _detect_host(argv, loaded_modules, available, environment)
    return host


def propagate_host(host: str, environment: MutableMapping[str, str]) -> None:
    """Persist an automatic selection for spawn children without changing config."""
    selected = _choice(
        host,
        variable=ACTIVE_HOST_ENV,
        allowed=_PROPAGATED_CHOICES,
    )
    environment[ACTIVE_HOST_ENV] = selected


def _is_hip() -> bool:
    try:
        import torch

        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


def _available_hosts() -> set[str]:
    available = set()
    for host in _HOSTS:
        try:
            if importlib.util.find_spec(host) is not None:
                available.add(host)
        except (ImportError, AttributeError, ValueError):
            continue
    return available


def _loaded_module_names() -> tuple[str, ...]:
    return tuple(sys.modules)


def install() -> None:
    """Install common guards and one host bridge at most once per process."""
    global _installed_pid

    pid = os.getpid()
    if _installed_pid == pid:
        return
    original_argv = getattr(sys, "orig_argv", ())
    if sys.argv == ["-c"] and "-c" in original_argv:
        index = original_argv.index("-c") + 1
        if index < len(original_argv) and re.fullmatch(
            r"from multiprocessing\.resource_tracker import main;main\([0-9]+\)",
            original_argv[index],
        ):
            _installed_pid = pid
            return
    if not _is_hip():
        _installed_pid = pid
        return

    launcher_args = list(original_argv)[-len(sys.argv)-1:]
    module_launch = (sys.argv[:1] == ["-m"] and len(launcher_args) >= 2
                     and launcher_args[0] == "-m"
                     and launcher_args[1] in {"sglang.launch_server", "hcu_sglang", "hcu_sglang.launch_server"}
                     and launcher_args[2:] == sys.argv[1:])
    cli_launch = bool(sys.argv) and PurePath(sys.argv[0]).name == "sglang" and sys.argv[1:2] == ["serve"]
    try:
        host, source = _detect_host(
            argv=["sglang", *sys.argv[1:]] if module_launch else sys.argv,
            loaded_modules=_loaded_module_names(),
            available=_available_hosts(),
            environment=os.environ,
        )
    except ValueError as exc:
        _installed_pid = pid
        LOG.warning("HCU adapter bootstrap disabled: %s", exc)
        return

    if host is None:
        _installed_pid = pid
        return

    if host == "sglang":
        from hcu_adapter.sglang_spawn_device import select_inherited_device

        plugins = os.environ.get("SGLANG_PLUGINS")
        plugin_names = {name.strip() for name in (plugins or "").split(",")}
        if (module_launch or cli_launch) and original_argv:
            from hcu_adapter.sglang_launch import apply_official_defaults

            try:
                apply_official_defaults(sys.argv[2:] if cli_launch else sys.argv[1:], original_argv)
            except (OSError, ValueError, RuntimeError) as exc:
                # sitecustomize would otherwise print the error and continue
                # launching a matching checkpoint without its required defaults.
                raise SystemExit(f"HCU launch configuration: {exc}") from exc
        if "hcu" in plugin_names and importlib.util.find_spec("hcu_sglang") is not None:
            from hcu_sglang.compat.aiter import bootstrap_aiter_compat

            bootstrap_aiter_compat()
        select_inherited_device()

    _installed_pid = pid
    propagate_host(host, os.environ)
    LOG.warning("HCU adapter bootstrap: host=%s source=%s", host, source)

    try:
        if host != "deepinfer":
            importlib.import_module("hcu_adapter.bridges.common").install()
    except Exception:
        LOG.warning("HCU adapter: common bridge not installed", exc_info=True)

    try:
        importlib.import_module(f"hcu_adapter.bridges.{host}").install()
    except Exception:
        LOG.warning("HCU adapter: %s bridge not installed", host, exc_info=True)
