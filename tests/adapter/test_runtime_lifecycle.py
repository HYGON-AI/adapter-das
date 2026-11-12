# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared CPU lifecycle contracts for independently named host consumers."""
from dataclasses import replace
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from hcu_adapter import extensions as ext
from hcu_adapter.runtime_lifecycle import ProcessLocalLifecycle
import hcu_adapter.runtime_lifecycle as lifecycle


CAP='shared-test-capability'


class ConsumerError(RuntimeError):
    pass


def activate(actions, checker):
    point=NS(name='owner',load=lambda: lambda: ext.ExtensionSpec(
        'owner',1,0,{CAP},lambda _: None,checker))
    def boot():
        context=ext.ExtensionContext('cpu',None,requested_capabilities={CAP})
        reports=ext.activate_extensions(context,[point],
            health_check_sink=actions.record_extension_health)
        actions.record_extension_reports(reports)
        actions.record_forward_extension_policy(context)
    actions.run_once(actions._runtime_action,boot)
    return actions.extension_reports()[0]


def fork_refusal(actions):
    reader,writer=os.pipe()
    pid=os.fork()
    if pid==0:
        os.close(reader)
        signal.alarm(4)
        try:
            assert not actions.extension_reports()
            actions.require_runtime_healthy()
        except ConsumerError:
            os.write(writer,b'REFUSED')
            os._exit(0)
        except BaseException:
            os._exit(2)
        os._exit(3)
    os.close(writer)
    try:
        ready,_,_=select.select([reader],[],[],6)
        if not ready:os.kill(pid,signal.SIGKILL)
        value=os.read(reader,32) if ready else b'TIMEOUT'
        _,status=os.waitpid(pid,0)
        assert value==b'REFUSED' and os.waitstatus_to_exitcode(status)==0
    finally:
        os.close(reader)


@pytest.fixture(params=['strict', 'requested'])
def health_guard(actions, request):
    return (lambda: actions.require_extension_healthy(CAP)) if request.param == 'strict' else actions.require_requested_extensions_healthy


@pytest.fixture(params=['sglang-runtime','vllm-runtime'])
def actions(request):
    return ProcessLocalLifecycle(runtime_action=request.param,error_type=ConsumerError)


def test_exact_report_duplicate_and_sticky_failure_use_same_contract_for_both_consumers(actions):
    calls=[]
    report=activate(actions,lambda: calls.append('checked'))
    actions.require_extension_healthy(CAP)
    assert calls==['checked']
    with pytest.raises(ext.ExtensionCommitError):
        actions.record_extension_health(report,lambda: calls.append('BAD'))
    actions.record_extension_reports((replace(report),))
    with pytest.raises(ConsumerError):actions.require_extension_healthy(CAP)
    assert calls==['checked']
    actions.record_extension_reports((report,))
    with pytest.raises(ConsumerError):actions.require_runtime_healthy()
    with pytest.raises(ConsumerError):actions.run_once(actions._runtime_action,lambda: calls.append('RETRY'))
    assert calls==['checked']


@pytest.mark.skipif(not hasattr(os,'fork'),reason='actual bounded CPU fork required')
def test_failure_rendering_forks_and_raises_without_losing_host_latch(actions, health_guard):
    observations=[]
    class BadString(RuntimeError):
        def __str__(self):
            fork_refusal(actions)
            observations.append('child refused')
            raise KeyboardInterrupt('diagnostic failed')
    def checker():raise BadString()
    activate(actions,checker)
    with pytest.raises(ConsumerError) as caught:health_guard()
    assert isinstance(caught.value.__cause__,BadString)
    assert observations==['child refused'] and actions._health_inflight==0
    with pytest.raises(ConsumerError):actions.require_runtime_healthy()


@pytest.mark.skipif(not hasattr(os,'fork'),reason='actual bounded CPU fork required')
def test_inflight_callback_fork_refuses_without_inherited_lock_deadlock(actions, health_guard):
    entered,release=threading.Event(),threading.Event()
    failures=[]
    def checker():
        entered.set()
        assert release.wait(8)
    activate(actions,checker)
    def check():
        try:health_guard()
        except BaseException as error:failures.append(error)
    thread=threading.Thread(target=check,daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        fork_refusal(actions)
    finally:
        release.set()
        thread.join(4)
    assert not thread.is_alive() and not failures
    actions.require_runtime_healthy()


def test_caught_checker_reentry_cannot_heal_either_consumer(actions, health_guard):
    calls=[]
    def checker():
        try:actions.run_once('other',lambda: calls.append('BAD'))
        except ext.ExtensionCommitError:pass
    activate(actions,checker)
    with pytest.raises(ConsumerError):health_guard()
    assert not calls
    with pytest.raises(ConsumerError):actions.require_runtime_healthy()


def test_consumer_instances_do_not_share_reports_or_failure_state():
    a=ProcessLocalLifecycle(runtime_action='vllm-runtime',error_type=ConsumerError)
    b=ProcessLocalLifecycle(runtime_action='sglang-runtime',error_type=ConsumerError)
    activate(a,lambda: (_ for _ in ()).throw(ValueError('owner drift')))
    activate(b,lambda: None)
    with pytest.raises(ConsumerError):a.require_extension_healthy(CAP)
    b.require_extension_healthy(CAP)
    b.require_runtime_healthy()
    assert a.extension_reports()[0] is not b.extension_reports()[0]


@pytest.mark.parametrize('host',['sglang','vllm'])
def test_actual_shared_and_bridge_imports_for_both_hosts_deny_frameworks_and_torch(host):
    # -S isolates the import contract from system .pth/sitecustomize hooks.
    # Explicit paths include the actual prebound source/installed package and
    # packaging dependency, while a finder refuses every framework/kernel load.
    root=str(Path(lifecycle.__file__).resolve().parent.parent)
    paths=[root]+[p for p in sys.path if p and p!=root]
    code='''
import importlib.abc, importlib, json, sys
sys.path[:0]=PATHS
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','vllm','sglang','sgl_kernel','hcu_vllm_das'}:
            raise AssertionError('forbidden framework import: '+fullname)
sys.meta_path.insert(0,Deny())
from hcu_adapter import runtime_lifecycle as shared
sg=importlib.import_module('hcu_adapter.bridges.sglang')
vl=importlib.import_module('hcu_adapter.bridges.vllm')
assert issubclass(vl.ProcessLocalActions,shared.ProcessLocalLifecycle)
assert callable(sg.install)
from hcu_adapter.extensions import ExtensionContext, ExtensionSpec, activate_extensions
from types import SimpleNamespace as NS
actions=shared.ProcessLocalLifecycle(runtime_action='isolated')
events=[]
context=ExtensionContext('cpu',None,requested_extensions={'owner'})
def boot():
    reports=activate_extensions(context,[NS(name='owner',load=lambda: lambda:
        ExtensionSpec('declared',1,0,{'cap'},lambda _: None,lambda: events.append('owner')))],
        health_check_sink=actions.record_extension_health)
    actions.record_extension_reports(reports)
    actions.record_forward_extension_policy(context)
actions.run_once('isolated',boot)
actions.require_requested_extensions_healthy()
assert events==['owner']
from pathlib import Path
for module in (shared,sg,vl):assert Path(module.__file__).is_relative_to(ROOT)
assert not {'torch','sglang','vllm','sgl_kernel','hcu_vllm_das'}.intersection(sys.modules)
print(json.dumps({'host':HOST,'shared':shared.__file__,'sglang_bridge':sg.__file__,'vllm_bridge':vl.__file__}))
'''.replace('PATHS',repr(paths)).replace('ROOT',repr(root)).replace('HOST',repr(host))
    env=dict(os.environ,HCU_ADAPTER_HOST=host,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-S','-B','-c',code],env=env,text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['host']==host


def enroll(actions, *, names=(), capabilities=(), owners=None):
    """Use real API1 activation; entry-point names deliberately differ from spec names."""
    context = ext.ExtensionContext('cpu', None, names, capabilities)
    if owners is None:
        owners = [('owner', {CAP}, lambda: None)]
    points = [NS(name=name, load=lambda name=name, caps=caps, checker=checker:
                 lambda: ext.ExtensionSpec('declared-' + name, 1, 0, caps,
                                           lambda _: None, checker))
              for name, caps, checker in owners]
    def boot():
        reports = ext.activate_extensions(context, points,
            health_check_sink=actions.record_extension_health)
        actions.record_extension_reports(reports)
        actions.record_forward_extension_policy(context)
    actions.run_once(actions._runtime_action, boot)
    return context


@pytest.mark.parametrize('names,capabilities', [({'owner'}, ()), ((), {CAP}), ({'owner'}, {CAP})])
def test_forward_freezes_name_capability_union_and_checks_each_owner_once(actions, names, capabilities):
    calls = []
    enroll(actions, names=names, capabilities=capabilities,
           owners=[('owner', {CAP}, lambda: calls.append('owner')),
                   ('optional', {'other'}, lambda: calls.append('BAD'))])
    assert calls == []  # Enrollment never runs an owner under run_once's lock.
    actions.require_requested_extensions_healthy()
    assert calls == ['owner']


def test_forward_accepts_explicit_no_checker_and_ignores_unrequested_optional_failure(actions):
    def failed(_):
        raise ValueError('clean optional preparation failure')
    context = ext.ExtensionContext('cpu', None, {'owner'})
    def boot():
        points = [NS(name='owner', load=lambda: lambda: ext.ExtensionSpec(
                    'declared-owner', 1, 0, {CAP}, lambda _: None)),
                  NS(name='optional', load=lambda: lambda: ext.ExtensionSpec(
                    'optional', 1, 0, {'other'}, failed))]
        reports = ext.activate_extensions(context, points, health_check_sink=actions.record_extension_health)
        actions.record_extension_reports(reports)
        actions.record_forward_extension_policy(context)
    actions.run_once(actions._runtime_action, boot)
    actions.require_requested_extensions_healthy()
    assert next(report for report in actions.extension_reports()
                if report.entry_point_name == 'optional').status is ext.ExtensionStatus.FAILED
    # The separate strict capability API still requires an actual checker.
    with pytest.raises(ConsumerError, match='exact current-process'):
        actions.require_extension_healthy(CAP)


def test_forward_accepts_completed_empty_policy_but_refuses_unknown_or_missing_enrollment(actions):
    empty = ProcessLocalLifecycle(runtime_action='empty', error_type=ConsumerError)
    enroll(empty, owners=[])
    empty.require_requested_extensions_healthy()
    # Latch-only API remains permissive before activation.
    actions.require_runtime_healthy()
    with pytest.raises(ConsumerError, match='completed|enroll'):
        actions.require_requested_extensions_healthy()
    unenrolled = ProcessLocalLifecycle(runtime_action='unenrolled', error_type=ConsumerError)
    unenrolled.run_once('unenrolled', lambda: None)
    with pytest.raises(ConsumerError, match='enroll'):
        unenrolled.require_requested_extensions_healthy()


def test_forward_refuses_incomplete_activation_without_invoking_checker(actions):
    calls = []
    context = ext.ExtensionContext('cpu', None, {'owner'})
    def boot():
        report = ext.ExtensionReport('owner', 'declared', ext.ExtensionStatus.ACTIVE, {CAP}, '')
        actions.record_extension_health(report, lambda: calls.append('BAD'))
        actions.record_extension_reports((report,))
        actions.record_forward_extension_policy(context)
        with pytest.raises(ConsumerError, match='completed'):
            actions.require_requested_extensions_healthy()
    with pytest.raises(ConsumerError, match='completed'):
        actions.run_once(actions._runtime_action, boot)
    assert calls == []


@pytest.mark.parametrize('change', ['same', 'equal', 'failed', 'duplicate', 'remove-readd',
                                  'lost-binding', 'rebound', 'wrong-pid'])
def test_forward_frozen_report_and_checker_cannot_be_replaced_or_resurrected(actions, change):
    calls = []
    enroll(actions, names={'owner'}, owners=[('owner', {CAP}, lambda: calls.append('BAD'))])
    report, = actions.extension_reports()
    if change == 'lost-binding':
        actions._health_bindings.clear()
    elif change in {'rebound', 'wrong-pid'}:
        binding = actions._health_bindings['owner']
        actions._health_bindings['owner'] = (report, binding[1],
                                            os.getpid() + (change == 'wrong-pid'))
    else:
        replacement = {'same': (report,), 'equal': (replace(report),),
                       'failed': (replace(report, status=ext.ExtensionStatus.FAILED),),
                       'duplicate': (report, report), 'remove-readd': ()}[change]
        actions.record_extension_reports(replacement)
        if change == 'remove-readd':
            actions.record_extension_reports((report,))
            actions.record_extension_health(report, lambda: calls.append('BAD'))
    with pytest.raises(ConsumerError):
        actions.require_requested_extensions_healthy()
    assert calls == []
    with pytest.raises(ConsumerError):
        actions.run_once(actions._runtime_action, lambda: None)


def test_forward_group_uses_one_generation_and_releases_host_lock_for_all_callbacks(actions):
    entered, changed = threading.Event(), threading.Event()
    calls = []
    def first():
        calls.append('first')
        entered.set()
        assert changed.wait(3), 'owner called under host lock'
    enroll(actions, names={'first', 'second'}, owners=[('first', {'a'}, first),
            ('second', {'b'}, lambda: calls.append('second'))])
    def publish():
        assert entered.wait(3)
        actions.record_extension_reports(actions.extension_reports())
        changed.set()
    worker = threading.Thread(target=publish, daemon=True)
    worker.start()
    try:
        with pytest.raises(ConsumerError, match='changed|generation'):
            actions.require_requested_extensions_healthy()
    finally:
        worker.join(4)
    assert not worker.is_alive() and calls == ['first', 'second']
    assert actions._health_inflight == 0


@pytest.mark.parametrize('fault', ['caught-reentry', 'baseexception', 'non-none', 'coroutine'])
def test_forward_checker_failures_latch_and_coroutine_is_closed(actions, fault):
    coroutines = []
    async def wrong_async():
        pytest.fail('coroutine must never execute')
    def checker():
        if fault == 'caught-reentry':
            try:
                actions.require_requested_extensions_healthy()
            except ext.ExtensionCommitError:
                return
        if fault == 'baseexception':
            raise KeyboardInterrupt('owner failure')
        if fault == 'coroutine':
            result = wrong_async()
            coroutines.append(result)
            return result
        return 1
    enroll(actions, names={'owner'}, owners=[('owner', {CAP}, checker)])
    with pytest.raises(ConsumerError):
        actions.require_requested_extensions_healthy()
    assert actions._health_inflight == 0
    if coroutines:
        assert coroutines[0].cr_frame is None
    with pytest.raises(ConsumerError):
        actions.require_runtime_healthy()


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='actual bounded CPU fork required')
def test_forward_healthy_fork_requires_fresh_child_runtime_enrollment(actions):
    enroll(actions, names={'owner'})
    actions.require_requested_extensions_healthy()
    pid = os.fork()
    if pid == 0:
        signal.alarm(4)
        try:
            assert not actions.extension_reports()
            # Do not call the guard before normal child activation: rejection
            # intentionally latches, and cannot then be repaired by activation.
            enroll(actions, names={'owner'})
            actions.require_requested_extensions_healthy()
        except BaseException:
            os._exit(2)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    pid = os.fork()
    if pid == 0:
        signal.alarm(4)
        try:
            actions.require_requested_extensions_healthy()
        except ConsumerError:
            os._exit(0)
        os._exit(3)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    actions.require_requested_extensions_healthy()


def test_forward_checks_disjoint_name_capability_union_with_immutable_input_sets(actions):
    names, caps, calls = {'a'}, {'b-cap'}, []
    enroll(actions, names=names, capabilities=caps, owners=[
        ('a', {'a-cap'}, lambda: calls.append('a')),
        ('b', {'b-cap'}, lambda: calls.append('b')),
        ('optional', {'other'}, lambda: calls.append('BAD'))])
    names.clear()
    caps.add('missing')
    actions.require_requested_extensions_healthy()
    assert calls == ['a', 'b']


@pytest.mark.parametrize('fault', ['missing-name', 'declared-name', 'failed-name',
                                  'duplicate-name', 'missing-cap', 'duplicate-cap', 'failed-cap'])
def test_forward_enrollment_requires_exact_unique_active_requested_owners(actions, fault):
    active = ext.ExtensionReport('entry', 'declared', ext.ExtensionStatus.ACTIVE, {'cap'}, '')
    failed = replace(active, status=ext.ExtensionStatus.FAILED)
    reports, names, caps = (active,), {'entry'}, ()
    if fault == 'missing-name':
        names = {'missing'}
    elif fault == 'declared-name':
        names = {'declared'}
    elif fault == 'failed-name':
        reports = (failed,)
    elif fault == 'duplicate-name':
        reports = (active, active)
    else:
        names, caps = (), {'cap'}
        if fault == 'missing-cap':
            caps = {'missing'}
        elif fault == 'duplicate-cap':
            reports = (active, replace(active, entry_point_name='second'))
        else:
            reports = (failed,)
    def boot():
        actions.record_extension_reports(reports)
        actions.record_forward_extension_policy(ext.ExtensionContext('cpu', None, names, caps))
    with pytest.raises(ConsumerError, match='exactly one ACTIVE'):
        actions.run_once(actions._runtime_action, boot)
    with pytest.raises(ConsumerError):
        actions.require_requested_extensions_healthy()


def test_forward_enrollment_cannot_be_changed_or_reentered_after_activation(actions):
    context = ext.ExtensionContext('cpu', None)
    def boot():
        report = ext.ExtensionReport('owner', 'declared', ext.ExtensionStatus.ACTIVE, {CAP}, '')
        actions.record_extension_reports((report,))
        actions.record_forward_extension_policy(context)
        actions.record_forward_extension_policy(context)  # Exact repeat is harmless.
        actions.record_forward_extension_policy(
            ext.ExtensionContext('cpu', None, requested_extensions={'owner'}))
    with pytest.raises(ConsumerError, match='enrollment changed'):
        actions.run_once(actions._runtime_action, boot)
    fresh = ProcessLocalLifecycle(runtime_action='fresh', error_type=ConsumerError)
    enroll(fresh, owners=[])
    with pytest.raises(ext.ExtensionCommitError, match='enrollment'):
        fresh.record_forward_extension_policy(context)
    fresh.require_requested_extensions_healthy()


@pytest.mark.parametrize('republication', [False, True])
def test_forward_no_checker_marker_cannot_adopt_later_binding_or_publication(actions, republication):
    enroll(actions, names={'owner'}, owners=[('owner', {CAP}, None)])
    report, = actions.extension_reports()
    if republication:
        actions.record_extension_reports((report,))
    else:
        actions.record_extension_health(report, lambda: None)
    with pytest.raises(ConsumerError):
        actions.require_requested_extensions_healthy()


def test_strict_capability_keeps_same_object_publication_semantics_separate_from_forward(actions):
    calls = []
    enroll(actions, names={'owner'}, owners=[('owner', {CAP}, lambda: calls.append('owner'))])
    actions.record_extension_reports(actions.extension_reports())
    actions.require_extension_healthy(CAP)
    assert calls == ['owner']
    with pytest.raises(ConsumerError, match='generation'):
        actions.require_requested_extensions_healthy()
    assert calls == ['owner']
