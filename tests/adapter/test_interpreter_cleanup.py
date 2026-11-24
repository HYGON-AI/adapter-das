# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Frozen-cycle finalization in real interpreters, fork and spawn.

Run with HOSTnone and hidden GPUs:
python -m pytest -q tests/adapter/test_interpreter_cleanup.py
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PROGRAM = r'''
import gc, os, sys
sys.path.insert(0, ROOT)
from hcu_adapter.infra.interpreter_cleanup import register_frozen_heap_cleanup

class Owner:
    def __init__(self):
        self.cycle = self
    def __del__(self, write=os.write):
        write(1, b'OWNER_FINALIZED\n')

if ENABLED:
    register_frozen_heap_cleanup()
    register_frozen_heap_cleanup()
owner = Owner()
gc.freeze()
if REFREEZE:
    gc.unfreeze()
    gc.collect()
    gc.freeze()
del owner
gc.collect()
assert gc.get_freeze_count() > 0
os.write(1, b'SERVING_FINISHED\n')
if FORK:
    child = os.fork()
    if child:
        assert os.waitpid(child, 0)[1] == 0
    else:
        register_frozen_heap_cleanup()
'''


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('refreeze', [False, True])
@pytest.mark.parametrize('fork', [False, True])
def test_frozen_owner_finalizes_only_after_serving(enabled, refreeze, fork):
    if fork and not hasattr(os, 'fork'):
        pytest.skip('requires fork')
    # A forked child always activates the adapter callback independently.
    expected = (2 if enabled else 1) if fork else int(enabled)
    program = ('ROOT = ' + repr(str(ROOT)) + '\nENABLED = ' + repr(enabled)
               + '\nREFREEZE = ' + repr(refreeze) + '\nFORK = ' + repr(fork)
               + '\n' + PROGRAM)
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', program],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('SERVING_FINISHED') == 1
    assert result.stdout.count('OWNER_FINALIZED') == expected, result.stdout
    if expected:
        assert result.stdout.index('SERVING_FINISHED') < result.stdout.index('OWNER_FINALIZED')


def test_cleanup_import_and_registration_are_framework_free():
    program = '''
import importlib.abc, sys, gc
sys.path.insert(0, ROOT)
class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'triton', 'vllm', 'sglang'}:
            raise AssertionError('unexpected framework import: ' + fullname)
sys.meta_path.insert(0, BlockFrameworks())
before = gc.get_freeze_count()
from hcu_adapter.infra.interpreter_cleanup import register_frozen_heap_cleanup
register_frozen_heap_cleanup()
register_frozen_heap_cleanup()
assert gc.get_freeze_count() == before
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c',
                             'ROOT = ' + repr(str(ROOT)) + '\n' + program],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('start_method', ['spawn', 'forkserver'])
def test_multiprocessing_child_installs_its_own_cleanup(tmp_path, start_method):
    import multiprocessing

    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip('unsupported process start method')
    script = tmp_path / 'spawn_cleanup.py'
    script.write_text('ROOT = ' + repr(str(ROOT)) + '\n' + '''
import gc, multiprocessing, os, sys
sys.path.insert(0, ROOT)

class Owner:
    def __init__(self):
        self.cycle = self
    def __del__(self, write=os.write):
        write(1, b'CHILD_OWNER_FINALIZED\\n')

def child():
    from hcu_adapter.infra.interpreter_cleanup import register_frozen_heap_cleanup
    register_frozen_heap_cleanup()
    owner = Owner()
    gc.freeze()
    del owner
    gc.collect()
    os.write(1, b'CHILD_SERVING_FINISHED\\n')

if __name__ == '__main__':
    from hcu_adapter.infra.interpreter_cleanup import register_frozen_heap_cleanup
    register_frozen_heap_cleanup()
    process = multiprocessing.get_context(sys.argv[1]).Process(target=child)
    process.start()
    process.join(20)
    assert not process.is_alive()
    assert process.exitcode == 0
''')
    result = subprocess.run([sys.executable, '-I', '-S', '-B', str(script), start_method],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count('CHILD_SERVING_FINISHED') == 1
    # multiprocessing's forkserver children use os._exit: ordinary atexit
    # callbacks are deliberately not an os._exit/SIGKILL cleanup guarantee.
    expected = 1 if start_method == 'spawn' else 0
    assert result.stdout.count('CHILD_OWNER_FINALIZED') == expected, result.stdout
