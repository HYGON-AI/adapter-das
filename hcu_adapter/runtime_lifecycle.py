# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free process-local lifecycle and extension-health transactions.

Host bridges supply their protected action ID and public error type. This
module owns locks, report binding, fork reset and failure publication together;
no framework admission, backend mapping or activation policy lives here.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from threading import RLock, local

from hcu_adapter.extensions import (
    ExtensionCommitError, ExtensionContext, ExtensionReport, ExtensionStatus,
)


_HealthBinding = tuple[ExtensionReport, Callable[[], None], int]


@dataclass(frozen=True)
class _ForwardExtensionPolicy:
    requested_extensions: frozenset[str]
    requested_capabilities: frozenset[str]
    owners: tuple[tuple[ExtensionReport, _HealthBinding | None], ...]
    pid: int
    generation: int


class ProcessLocalLifecycle:
    """Run required callbacks once per PID and latch partial failures."""

    def __init__(self, *, runtime_action: str, error_type: type[RuntimeError] = RuntimeError) -> None:
        if not isinstance(runtime_action, str) or not runtime_action:
            raise TypeError("runtime_action must be a non-empty string")
        if not isinstance(error_type, type) or not issubclass(error_type, RuntimeError):
            raise TypeError("error_type must be a RuntimeError subclass")
        self._runtime_action = runtime_action
        self._error_type = error_type
        self._owner_pid = os.getpid()
        self._lock = RLock()
        self._records: dict[str, str | None] = {}
        self._running: set[str] = set()
        self._extension_reports: tuple[ExtensionReport, ...] = ()
        self._health_bindings: dict[str, _HealthBinding] = {}
        self._forward_policy: _ForwardExtensionPolicy | None = None
        self._report_generation = 0
        self._health_inflight = 0
        self._health_context = local()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork_child)

    def _reset(self, owner_pid: int) -> None:
        # Only the configured runtime action inherits failure. Its registry
        # mutations also survive fork. Other actions keep their old reset.
        failure = self._records.get(self._runtime_action)
        if failure is None and self._runtime_action in self._running:
            failure = f"inherited incomplete {self._runtime_action} activation; spawn a fresh process"
        if failure is None and self._health_inflight:
            failure = "inherited incomplete extension health check; spawn a fresh process"
        self._owner_pid = owner_pid
        self._lock = RLock()
        self._records = {} if failure is None else {self._runtime_action: failure}
        self._running = set()
        self._extension_reports = ()
        self._health_bindings = {}
        self._forward_policy = None
        self._report_generation = 0
        self._health_inflight = 0
        self._health_context = local()

    def _after_fork_child(self) -> None:
        self._reset(os.getpid())

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if pid != self._owner_pid:
            self._reset(pid)

    def run_once(self, action_id: str, callback: Callable[[], object]) -> None:
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            if action_id in self._records:
                failure = self._records[action_id]
                if failure is not None:
                    raise self._error_type(failure)
                return
            if action_id == self._runtime_action and action_id in self._running:
                raise self._error_type(f"recursive {self._runtime_action} activation")
            self._running.add(action_id)
            try:
                callback()
            except Exception as exc:
                # Diagnostic formatting can raise, re-enter or fork. Publish
                # refusal before it runs and before finally clears _running.
                failure = "runtime action failed; exception details unavailable"
                self._records[action_id] = failure
                try:
                    failure = f"{type(exc).__name__}: {exc}"
                except BaseException:
                    pass
                self._records[action_id] = failure
                raise self._error_type(failure) from exc
            finally:
                self._running.discard(action_id)
            if self._records.get(action_id) is not None:
                raise self._error_type(self._records[action_id])
            self._records[action_id] = None

    def require_runtime_healthy(self) -> None:
        """Read failure only; safe/reentrant inside the runtime callback."""
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            failure = self._records.get(self._runtime_action)
            if failure is not None:
                raise self._error_type(failure)

    def record_extension_reports(
        self,
        reports: tuple[ExtensionReport, ...],
    ) -> None:
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            self._extension_reports = tuple(reports)
            self._report_generation += 1
            # A value-equal replacement report must not borrow a prior checker.
            # Discard stale bindings permanently; no later report can resurrect it.
            self._health_bindings = {
                name: binding for name, binding in self._health_bindings.items()
                if binding[2] == self._owner_pid
                and sum(report.entry_point_name == name for report in reports) == 1
                and any(report is binding[0] and report.status is ExtensionStatus.ACTIVE
                        for report in reports)
            }

    def _reject_health_reentry(self) -> None:
        # Never acquire a host lock from an owner checker. The outer call will
        # latch after the owner lock has unwound, even if the owner caught this.
        if getattr(self._health_context, "checking", False):
            self._health_context.violation = True
            raise ExtensionCommitError("health checker must not re-enter host lifecycle")

    def record_extension_health(self, report: ExtensionReport, checker: Callable[[], None]) -> None:
        """Activation sink: bind exact report/checker/PID, not backend names."""
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            if report.status is not ExtensionStatus.ACTIVE or not callable(checker):
                raise ExtensionCommitError("invalid ACTIVE extension health binding")
            if report.entry_point_name in self._health_bindings:
                raise ExtensionCommitError("duplicate extension health binding")
            self._health_bindings[report.entry_point_name] = (report, checker, self._owner_pid)

    def record_forward_extension_policy(self, context: ExtensionContext) -> None:
        """Freeze requirements at the trusted runtime publication boundary.

        No checker runs here: run_once still holds the outer activation lock.
        None is an explicit API1 no-checker marker, never inferred at forward.
        """
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            self.require_runtime_healthy()
            if self._runtime_action not in self._running:
                raise ExtensionCommitError("forward policy requires runtime activation enrollment")
            names = frozenset(context.requested_extensions)
            capabilities = frozenset(context.requested_capabilities)
            selected = []
            for required, attribute, label in (
                (names, "entry_point_name", "extension"),
                (capabilities, "capabilities", "capability"),
            ):
                for name in sorted(required):
                    owners = [report for report in self._extension_reports
                              if (report.entry_point_name == name if attribute == "entry_point_name"
                                  else name in report.capabilities)]
                    if len(owners) != 1 or owners[0].status is not ExtensionStatus.ACTIVE:
                        raise ExtensionCommitError(
                            f"{label} {name!r} requires exactly one ACTIVE extension owner")
                    if not any(report is owners[0] for report in selected):
                        selected.append(owners[0])
            owners = tuple((report, self._health_bindings.get(report.entry_point_name))
                           for report in selected)
            for report, binding in owners:
                self._validate_health_binding(report, binding, allow_absent=True)
            policy = self._forward_policy
            if policy is not None:
                if (policy.pid != self._owner_pid or policy.generation != self._report_generation
                        or policy.requested_extensions != names
                        or policy.requested_capabilities != capabilities
                        or len(policy.owners) != len(owners)
                        or any(old_report is not report or old_binding is not binding
                               for (old_report, old_binding), (report, binding)
                               in zip(policy.owners, owners))):
                    raise ExtensionCommitError("forward extension policy enrollment changed")
                return
            self._forward_policy = _ForwardExtensionPolicy(
                names, capabilities, owners, self._owner_pid, self._report_generation)

    def _validate_health_binding(
        self, report: ExtensionReport, binding: _HealthBinding | None, *, allow_absent: bool,
    ) -> None:
        if ((binding is None and not allow_absent)
                or (binding is not None and (binding[0] is not report
                                            or binding[2] != os.getpid()))):
            raise ExtensionCommitError("extension has no exact current-process health binding")

    def _requested_health_snapshot(self) -> tuple[tuple[ExtensionReport, _HealthBinding | None], ...]:
        if self._runtime_action not in self._records:
            raise ExtensionCommitError("forward health requires completed runtime activation")
        policy = self._forward_policy
        if policy is None or policy.pid != os.getpid():
            raise ExtensionCommitError("forward health requires current-process policy enrollment")
        if policy.generation != self._report_generation:
            raise ExtensionCommitError("forward extension policy report generation changed")
        for report, binding in policy.owners:
            if (report.status is not ExtensionStatus.ACTIVE
                    or sum(item.entry_point_name == report.entry_point_name
                           for item in self._extension_reports) != 1
                    or not any(item is report for item in self._extension_reports)
                    or self._health_bindings.get(report.entry_point_name) is not binding):
                raise ExtensionCommitError("forward extension report/health binding changed")
            self._validate_health_binding(report, binding, allow_absent=True)
        return policy.owners

    def require_requested_extensions_healthy(self) -> None:
        """Check the frozen requested-owner union after completed PID activation."""
        self._require_extensions_healthy(requested=True)

    def require_extension_healthy(self, capability: str) -> None:
        """Require a unique owner with an exact callable current-PID binding."""
        self._require_extensions_healthy(capability)

    def _require_extensions_healthy(self, capability: str | None = None, *, requested: bool = False) -> None:
        """One host snapshot, unlocked owner callbacks, then host verification.

        Both strict capability and frozen forward policy share failure/reentry,
        inflight and fork semantics. Arbitrary registry writers have no CAS.
        """
        self.require_runtime_healthy()
        inflight = False
        owner_pid = os.getpid()
        try:
            with self._lock:
                # Cover validation, owner callback, snapshot verification AND
                # failure publication. Fork must see inflight or a latched
                # failure until the complete transaction has finished.
                self._health_inflight += 1
                inflight = True
                if self._running:
                    raise ExtensionCommitError("extension health requires completed host activation")
                if requested:
                    snapshot = self._requested_health_snapshot()
                else:
                    owners = [report for report in self._extension_reports
                              if capability in report.capabilities]
                    if len(owners) != 1 or owners[0].status is not ExtensionStatus.ACTIVE:
                        raise ExtensionCommitError(
                            f"capability {capability!r} requires exactly one ACTIVE extension owner")
                    report = owners[0]
                    binding = self._health_bindings.get(report.entry_point_name)
                    self._validate_health_binding(report, binding, allow_absent=False)
                    snapshot = ((report, binding),)
                generation = self._report_generation
            self._health_context.checking = True
            self._health_context.violation = False
            try:
                for report, binding in snapshot:
                    if binding is None:
                        continue  # Only the frozen policy admits explicit no-checker.
                    result = binding[1]()
                    if result is not None:
                        # Close a native coroutine without running owner code.
                        import inspect
                        if inspect.iscoroutine(result):
                            result.close()
                        raise ExtensionCommitError("health checker must return None synchronously")
                    if self._health_context.violation:
                        raise ExtensionCommitError("health checker caught forbidden host lifecycle reentry")
            finally:
                self._health_context.checking = False
            with self._lock:
                if (owner_pid != os.getpid() or generation != self._report_generation
                        or any(self._health_bindings.get(report.entry_point_name) is not binding
                               for report, binding in snapshot)):
                    raise ExtensionCommitError("extension reports/health binding changed during health check")
                if requested:
                    self._requested_health_snapshot()
                self.require_runtime_healthy()
        except BaseException as exc:
            # Owner locks have unwound before the host latch is acquired.
            published = False
            with self._lock:
                failure = self._records.get(self._runtime_action)
                if failure is None:
                    # Publish refusal BEFORE arbitrary exception __str__ code.
                    # Formatting can raise, re-enter, block or fork. A child
                    # must inherit a failure even at that exact boundary.
                    failure = "extension health failed; exception details unavailable"
                    self._records[self._runtime_action] = failure
                    published = True
            if published:
                try:
                    detail = f"extension health failed: {type(exc).__name__}: {exc}"
                except BaseException:
                    detail = failure
                with self._lock:
                    self._records[self._runtime_action] = detail
                    failure = detail
            raise self._error_type(failure) from exc
        finally:
            # On failure, the except block has already published the host
            # latch. Decrementing earlier creates a fork-to-healthy window.
            if inflight and os.getpid() == owner_pid:
                with self._lock:
                    self._health_inflight -= 1

    def extension_reports(self) -> tuple[ExtensionReport, ...]:
        self._ensure_process_local()
        self._reject_health_reentry()
        with self._lock:
            return self._extension_reports


