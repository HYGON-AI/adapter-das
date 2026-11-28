# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""A CUDA-clean assertion belongs to a test file that is given a clean process.

``assert not torch.cuda.is_initialized()`` only holds in an interpreter that
never touched a device. Inside the L1 kernels segment one pytest process
hosts sglang and has CUDA up from the device tests, so such a file errors at
teardown (adapter-das#214: 25 errors in the 2026-11-28 nightly) unless the
conftest reruns it in the clean CPU interpreter or scopes it to the vllm lane.
"""

import ast
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _conftest():
    spec = importlib.util.spec_from_file_location(
        "adapter_conftest_under_test", HERE / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collected():
    """Files pytest collects under tests/adapter: test_*.py and cases/vllm_*.py."""
    for path in sorted(HERE.rglob("*.py")):
        if path.name.startswith("test_") or (
                path.parent.name == "cases" and path.name.startswith("vllm_")):
            yield path


def _is_the_call(node):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "is_initialized"
            and ast.unparse(node.func.value) == "torch.cuda")


def _asserts_cuda_clean(tree):
    """``assert not torch.cuda.is_initialized()`` (also ``is False`` / ``== False``)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and _is_the_call(test.operand)):
            return True
        if (isinstance(test, ast.Compare) and _is_the_call(test.left)
                and all(isinstance(op, (ast.Is, ast.Eq)) for op in test.ops)
                and any(isinstance(c, ast.Constant) and c.value is False
                        for c in test.comparators)):
            return True
    return False


def test_cuda_clean_assertions_get_a_clean_process():
    conftest = _conftest()
    unscoped = [
        path.relative_to(HERE).as_posix() for path in _collected()
        if _asserts_cuda_clean(ast.parse(path.read_text(encoding="utf-8")))
        and path.name not in conftest._CPU_RERUN_FILES
        and not conftest._is_vllm_host_file(str(path))]
    assert not unscoped, (
        f"{unscoped} assert a CUDA-clean process, but the L1 kernels segment "
        "is not one: list the file in conftest._CPU_RERUN_FILES and give its "
        "tests a clean rerun, or scope them to the vllm lane")


def test_clean_rerun_files_exist_and_rerun_themselves():
    conftest = _conftest()
    broken = []
    for name in sorted(conftest._CPU_RERUN_FILES):
        path = HERE / name
        if not path.is_file():
            broken.append(f"{name}: no such file")
            continue
        # When the clean batch fails, conftest leaves the in-process copies
        # alone: each test must then rerun itself to show the real failure.
        imported = {alias.name for node in ast.walk(
                        ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(node, ast.ImportFrom)
                    and node.module == "_cpu_process" for alias in node.names}
        if not imported & {"rerun", "rerun_if_host_loaded"}:
            broken.append(f"{name}: no clean rerun from _cpu_process")
    assert not broken, "\n".join(broken)
