# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual collection hook in fresh CPU-only pytest processes."""
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.parametrize("scenario,ran", [
    ("hidden", True),
    ("minus-one", True),
    ("missing-visibility", False),
    ("sglang-host", False),
    ("sglang-imported", False),
    ("initialized", False),
])
def test_cpu_contracts_execute_only_in_the_documented_clean_boundary(tmp_path, scenario, ran):
    hook = Path(__file__).with_name("conftest.py")
    (tmp_path / "conftest.py").write_bytes(hook.read_bytes())
    proof = tmp_path / "executed"
    (tmp_path / "test_vllm_cpu_probe.py").write_text(
        "from pathlib import Path\n"
        "def test_cpu_probe():\n"
        f"    Path({str(proof)!r}).write_text('executed')\n")
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("HCU_", "_HCU_", "VLLM_", "PYTHON",
                                          "HIP_", "CUDA_", "ROCR_"))}
    environment.update(HCU_ADAPTER_HOST="none", HIP_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    setup = ""
    if scenario == "minus-one":
        for key in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            environment[key] = "-1"
    elif scenario == "missing-visibility":
        environment.pop("ROCR_VISIBLE_DEVICES")
    elif scenario == "sglang-host":
        environment["_HCU_ADAPTER_ACTIVE_HOST"] = "sglang"
    elif scenario == "sglang-imported":
        setup = "import types;sys.modules['sglang']=types.ModuleType('sglang');"
    elif scenario == "initialized":
        # A collection-state double models a previously initialized parent;
        # no real device is touched and production CUDA guards remain intact.
        setup = ("import types;sys.modules['torch']=types.SimpleNamespace("
                 "cuda=types.SimpleNamespace(is_initialized=lambda:True));")
    # Do not run an installed Adapter sitecustomize before the test scenario.
    # The subprocess owns its isolation and uses only existing dependency paths.
    paths = [path for path in sys.path if "site-packages" in path or "dist-packages" in path]
    code = ("import sys;sys.path[:0]=" + repr(paths) + ";" + setup +
            "import pytest;raise SystemExit(pytest.main(sys.argv[1:]))")
    report = tmp_path / "result.xml"
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code,
        str(tmp_path / "test_vllm_cpu_probe.py"), "-q", "--junitxml=" + str(report)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == 1
    assert proof.exists() is ran
    assert (cases[0].find("skipped") is None) is ran
