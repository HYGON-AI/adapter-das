# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Packaging contracts for the dual-dist build (governance §7.4).

Two wheels, independent version lines, one interlock:
  sgl-kernel  = shadow kernel only (sgl_kernel/* incl. diffusion_rope_csrc JIT
                headers; no sitecustomize, no entry-points); version single
                source: sgl_kernel/version.py
  hcu-adapter = hcu_adapter/* (incl. launch_configs/*.yaml built-in presets) +
                sitecustomize.py + vllm entry-points;
                version single source: packaging/hcu-adapter/VERSION;
                Requires-Dist: sgl-kernel==<sgl_kernel version>

The release line used to ship one wheel that bundled hcu_adapter/; these
contracts pin the split so `packaging/build.sh` (nightly) and `pip wheel .`
(tests/dev) cannot drift back to a single dist. Non-.py data files are part of
the contract: each dist declares its own [tool.setuptools.package-data],
because the adapter build root cannot discover files outside itself.
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10, supported by the package.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = REPO_ROOT / "packaging" / "hcu-adapter"
ADAPTER_PYPROJECT = ADAPTER_ROOT / "pyproject.toml"


def _kernel_public_version():
    src = (REPO_ROOT / "sgl_kernel" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    assert m, "sgl_kernel/version.py must carry __version__ (single source)"
    return m.group(1).split("+", 1)[0]


def _build_wheel(source, wheel_dir, cwd):
    env = os.environ.copy()
    env["HCU_ADAPTER_HOST"] = "none"
    # no stamp/platform: both dists fall back to their public versions, so the
    # assertions below do not depend on the build time or the DTK label
    env.pop("DAS_BUILD_STAMP", None)
    env.pop("DAS_PLATFORM_TAG", None)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def _copy_repo(tmp_path):
    source = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist", "*.egg-info"
        ),
    )
    return source


def _members(wheel):
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist()
                    if n.endswith(".dist-info/METADATA"))
        return archive.read(name).decode("utf-8")


def test_kernel_wheel_is_kernel_only(tmp_path):
    source = _copy_repo(tmp_path)
    wheel = _build_wheel(source, tmp_path / "wheel-kernel", tmp_path)
    members = _members(wheel)

    assert wheel.name.startswith("sgl_kernel-"), wheel.name
    assert any(name.startswith("sgl_kernel/") for name in members)
    assert not [n for n in members if n.startswith("hcu_adapter/")], \
        "sgl-kernel must not bundle hcu_adapter/"
    assert "sitecustomize.py" not in members, \
        "sitecustomize belongs to the hcu-adapter dist"
    assert not [n for n in members if n.endswith(".dist-info/entry_points.txt")], \
        "vllm entry-points belong to the hcu-adapter dist"

    # data files stay with their dist: the JIT compiles these headers from the
    # installed package, so dropping the kernel's package-data breaks JIT
    headers = {
        f"sgl_kernel/diffusion_rope_csrc/{path.name}"
        for path in (REPO_ROOT / "sgl_kernel" / "diffusion_rope_csrc").glob("*.cuh")
    }
    assert headers, "expected sgl_kernel/diffusion_rope_csrc/*.cuh in the tree"
    assert not (headers - members), \
        f"kernel wheel is missing JIT headers: {sorted(headers - members)}"


def test_adapter_wheel_carries_adapter_sitecustomize_and_entry_points(tmp_path):
    wheel = _build_wheel(ADAPTER_ROOT, tmp_path / "wheel-adapter", tmp_path)
    members = _members(wheel)

    assert wheel.name.startswith("hcu_adapter-"), wheel.name
    assert any(name.startswith("hcu_adapter/") for name in members)
    assert "sitecustomize.py" in members
    assert not [n for n in members if n.startswith("sgl_kernel/")], \
        "hcu-adapter must not ship the kernel package"

    expected = {"sitecustomize.py"}
    for path in (REPO_ROOT / "hcu_adapter").rglob("*.py"):
        expected.add(path.relative_to(REPO_ROOT).as_posix())
    assert not (expected - members), f"adapter wheel is missing: {sorted(expected - members)}"

    # built-in launch configs are runtime data, loaded as
    # HCU_CONFIG=builtin:<name> (docs/env-vars.md); they only reach the wheel
    # through the adapter dist's explicit [tool.setuptools.package-data], since
    # package-dir points outside this build root
    configs = {
        f"hcu_adapter/launch_configs/{path.name}"
        for path in (REPO_ROOT / "hcu_adapter" / "launch_configs").glob("*.yaml")
    }
    assert configs, "expected hcu_adapter/launch_configs/*.yaml in the tree"
    assert not (configs - members), \
        f"adapter wheel is missing built-in launch configs: {sorted(configs - members)}"

    entry_points = [n for n in members if n.endswith(".dist-info/entry_points.txt")]
    assert entry_points, "adapter wheel must register the vllm entry-points"


def test_adapter_pins_the_kernel_public_version(tmp_path):
    wheel = _build_wheel(ADAPTER_ROOT, tmp_path / "wheel-pin", tmp_path)
    requirements = [
        line.split(":", 1)[1].strip()
        for line in _metadata(wheel).splitlines()
        if line.lower().startswith("requires-dist:")
    ]
    kernel_pins = [r for r in requirements if r.split(";")[0].strip().startswith("sgl-kernel")]
    expected = f"sgl-kernel=={_kernel_public_version()}"
    assert kernel_pins, f"adapter must pin the kernel, got {requirements}"
    assert kernel_pins[0].split(";")[0].strip() == expected, (kernel_pins, expected)


def test_vllm_official_entry_points_share_one_atomic_allowlist_name():
    project = tomllib.loads(ADAPTER_PYPROJECT.read_text(encoding="utf-8"))["project"]

    entry_points = project["entry-points"]
    assert entry_points["vllm.platform_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_platform_plugin"
    }
    assert entry_points["vllm.general_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
    }


def test_root_pyproject_ships_only_the_kernel():
    project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert project["project"]["name"] == "sgl-kernel"
    assert "entry-points" not in project["project"], \
        "root project must not declare vllm entry-points (they live on hcu-adapter)"
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["sgl_kernel*"]
