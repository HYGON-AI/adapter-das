# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dual-dist packaging contracts (governance §7.4).

Two wheels, independent version lines, one interlock:
  sgl-kernel  = pure shadow kernel (sgl_kernel/* only; no sitecustomize,
                no entry-points); version single source:
                sgl_kernel/version.py (kernel-line pairing)
  hcu-adapter = hcu_adapter/* + sitecustomize.py + vllm entry-points;
                version single source: packaging/hcu-adapter/VERSION
                (sglang-line pairing); interlocks the kernel dist via
                Requires-Dist: sgl-kernel==<sgl_kernel version>
"""

import os
import base64
import csv
import email
import hashlib
import importlib.util
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10, supported by the package.
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PYPROJECT = REPO_ROOT / "packaging" / "hcu-adapter" / "pyproject.toml"


def _kernel_version():
    src = (REPO_ROOT / "sgl_kernel" / "version.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    assert m, "sgl_kernel/version.py must carry __version__ (single source)"
    return m.group(1)


def _adapter_version():
    return (ADAPTER_PYPROJECT.parent / "VERSION").read_text(
        encoding="utf-8").strip()


def _build_wheel(source_dir, tmp_path, tag, stamp=None):
    wheel_dir = tmp_path / ("wheel-" + tag)
    wheel_dir.mkdir()
    env = os.environ.copy()
    env["HCU_ADAPTER_HOST"] = "none"
    env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", PIP_NO_INDEX="1")
    env.pop("DAS_BUILD_STAMP", None)
    if stamp is not None:
        env["DAS_BUILD_STAMP"] = stamp
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
            str(source_dir),
        ],
        check=True,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return next(wheel_dir.glob("*.whl"))


def _copy_repo(tmp_path):
    source = tmp_path / "repo"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".pytest_cache", "__pycache__", "build", "dist",
            "*.egg-info",
        ),
    )
    return source


@pytest.fixture(scope="module")
def kernel_wheel(tmp_path_factory):
    return _build_wheel(_copy_repo(tmp_path_factory.mktemp("src-kernel")),
                        tmp_path_factory.getbasetemp(), "kernel")


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory):
    # the adapter dist builds from packaging/hcu-adapter with sources left
    # at the repo root (src-layout mapping, nothing copied)
    source = _copy_repo(tmp_path_factory.mktemp("src-adapter"))
    return _build_wheel(source / "packaging" / "hcu-adapter",
                        tmp_path_factory.getbasetemp(), "adapter")


def _members(wheel):
    archive = zipfile.ZipFile(wheel)
    return set(archive.namelist()), archive


def test_kernel_wheel_is_pure_shadow_kernel(kernel_wheel):
    members, archive = _members(kernel_wheel)
    version = _kernel_version()

    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "sgl_kernel").rglob("*.py")
    }
    assert not (expected - members), f"kernel wheel missing: {sorted(expected - members)}"
    # purity: no adapter payload rides the kernel dist
    assert "sitecustomize.py" not in members
    assert not any(m.startswith("hcu_adapter/") for m in members), \
        "kernel wheel must not ship hcu_adapter/"
    assert f"sgl_kernel-{version}.dist-info/METADATA" in members
    metadata = archive.read(f"sgl_kernel-{version}.dist-info/METADATA").decode()
    assert "Requires-Dist: torch" in metadata  # torch/triton/packaging stay
    ep = f"sgl_kernel-{version}.dist-info/entry_points.txt"
    assert ep not in members, "kernel dist must not carry vllm entry-points"
    archive.close()


def test_adapter_wheel_carries_bootstrap_bridges_and_entry_points(adapter_wheel):
    members, archive = _members(adapter_wheel)
    version = _adapter_version()

    expected = {"sitecustomize.py"}
    expected.update(
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "hcu_adapter").rglob("*.py")
    )
    assert not (expected - members), f"adapter wheel missing: {sorted(expected - members)}"
    # purity: the kernel package does not ride the adapter dist
    assert not any(m.startswith("sgl_kernel/") for m in members), \
        "adapter wheel must not ship sgl_kernel/ (interlock pins the dist)"

    dist_info = f"hcu_adapter-{version}.dist-info"
    assert f"{dist_info}/METADATA" in members
    metadata = archive.read(f"{dist_info}/METADATA").decode()
    requirements = [Requirement(s) for s in email.message_from_string(metadata).get_all(
        "Requires-Dist", [])]
    assert [str(r.specifier) for r in requirements if r.name == "sgl-kernel"] == [
        "==" + _kernel_version().split("+", 1)[0]], \
        "adapter must interlock the kernel dist with an exact pin"
    assert "sgl-kernel~=" not in metadata

    entry_points = archive.read(f"{dist_info}/entry_points.txt").decode()
    assert "[vllm.platform_plugins]" in entry_points
    assert "[vllm.general_plugins]" in entry_points
    assert "hcu_adapter = hcu_adapter.bridges.vllm:hcu_platform_plugin" in entry_points
    assert "hcu_adapter = hcu_adapter.bridges.vllm:hcu_runtime_plugin" in entry_points
    archive.close()


def test_vllm_entry_points_declared_only_in_adapter_dist():
    adapter_project = tomllib.loads(
        ADAPTER_PYPROJECT.read_text(encoding="utf-8")
    )["project"]
    entry_points = adapter_project["entry-points"]
    assert entry_points["vllm.platform_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_platform_plugin"
    }
    assert entry_points["vllm.general_plugins"] == {
        "hcu_adapter": "hcu_adapter.bridges.vllm:hcu_runtime_plugin"
    }
    # entry-points migrated (not duplicated): the kernel dist declares none
    kernel_project = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert "entry-points" not in kernel_project


def test_interlock_pin_matches_single_version_source(adapter_wheel):
    version = _kernel_version().split("+", 1)[0]
    project = tomllib.loads(ADAPTER_PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert "dependencies" in project["dynamic"]
    assert "dependencies" not in project
    metadata = _wheel_metadata(adapter_wheel)
    pins = [Requirement(s) for s in metadata.get_all("Requires-Dist", [])
            if Requirement(s).name == "sgl-kernel"]
    assert len(pins) == 1 and str(pins[0].specifier) == f"=={version}"


def test_adapter_version_file_is_the_adapter_single_source(adapter_wheel):
    # packaging/hcu-adapter/VERSION is the adapter dist's own single source
    # (independent of sgl-kernel's line); the built wheel must carry it
    # verbatim in its metadata.
    members, archive = _members(adapter_wheel)
    dist_info = f"hcu_adapter-{_adapter_version()}.dist-info"
    metadata = archive.read(f"{dist_info}/METADATA").decode()
    archive.close()
    assert f"Version: {_adapter_version()}" in metadata


def _wheel_metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(archive.read(member))


@pytest.fixture(scope="module")
def stamped_pairs(tmp_path_factory):
    pairs = {}
    for profile, suffix in [("torch2130", ""), ("torch2110", ".torch2110")]:
        temp = tmp_path_factory.mktemp(profile)
        source = _copy_repo(temp)
        stamp = "2611111200.g23f605" + suffix
        pairs[profile] = (
            _build_wheel(source, temp, "kernel", stamp),
            _build_wheel(source / "packaging/hcu-adapter", temp, "adapter", stamp),
        )
    return pairs


@pytest.mark.parametrize("profile,torch_version,triton_version", [
    ("torch2130", "2.13.0", "3.7.0"), ("torch2110", "2.11.0", "3.6.0"),
])
def test_real_stamped_pair_dependency_profile(stamped_pairs, profile, torch_version,
                                              triton_version):
    kernel, adapter = stamped_pairs[profile]
    km, am = _wheel_metadata(kernel), _wheel_metadata(adapter)
    base = {r.name: r for s in am.get_all("Requires-Dist", [])
            if (r := Requirement(s)).marker is None}
    assert str(base["torch"].specifier) == "==" + torch_version
    assert str(base["triton"].specifier) == "==" + triton_version
    assert str(base["packaging"].specifier) == ">=23"
    assert str(base["sgl-kernel"].specifier) == "==" + km["Version"]
    assert am["Version"].split("+", 1)[1] == km["Version"].split("+", 1)[1]
    if profile == "torch2110":
        assert am["Version"].endswith(".torch2110")
    for wheel in [kernel, adapter]:
        with zipfile.ZipFile(wheel) as archive:
            record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
            for name, digest, size in csv.reader(io.StringIO(archive.read(record).decode())):
                if digest:
                    data = archive.read(name)
                    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
                    assert digest == "sha256=" + encoded.decode()
                    assert int(size) == len(data)
    # A profile must not change any runtime source or extra/entry-point declaration.
    with zipfile.ZipFile(adapter) as archive:
        for name in archive.namelist():
            if name.startswith("hcu_adapter/") or name == "sitecustomize.py":
                assert archive.read(name) == (REPO_ROOT / name).read_bytes()
    assert set(am.get_all("Provides-Extra", [])) == set(
        _wheel_metadata(stamped_pairs["torch2130"][1]).get_all("Provides-Extra", []))


@pytest.mark.parametrize("stamp", [
    "2611111200.g23f605.torch2150", "2611111200.torch2110.g23f605",
    "2611111200.g23f605.torch2110.torch2110",
    "2611111200.g23f605.torch2130.torch2110",
    "2611111200.g23f605.TORCH2110", "2611111200.g23f605-torch2110",
    "2611111200.g23f605_torch2110",
])
def test_backend_rejects_invalid_reserved_profile(stamp, monkeypatch):
    spec = importlib.util.spec_from_file_location("packaging_pin_test", ADAPTER_PYPROJECT.parent / "_pin.py")
    pin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pin)
    monkeypatch.setenv("DAS_BUILD_STAMP", stamp)
    with pytest.raises(ValueError, match="profile"):
        pin.base_dependencies()


def test_build_cli_profiles_and_explicit_stamp_conflict(tmp_path):
    source = _copy_repo(tmp_path)
    env = {**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
           "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1", "HIP_VISIBLE_DEVICES": "",
           "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
           "DAS_BUILD_STAMP": "2611111200.g23f605"}
    result = subprocess.run(["bash", "packaging/build.sh", "--runtime-profile", "torch2110"],
                            cwd=source, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    adapter = next((source / "dist").glob("hcu_adapter-*.whl"))
    assert _wheel_metadata(adapter)["Version"].endswith(".torch2110")
    env["DAS_BUILD_STAMP"] += ".torch2110"
    result = subprocess.run(["bash", "packaging/build.sh", "--runtime-profile", "torch2130"],
                            cwd=source, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "conflict" in result.stderr.lower()


def test_nondefault_unstamped_build_is_rejected():
    env = {**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable}
    env.pop("DAS_BUILD_STAMP", None)
    result = subprocess.run(["bash", "packaging/build.sh", "--no-stamp", "--runtime-profile", "torch2110"],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "no-stamp" in result.stderr and "torch2110" in result.stderr


@pytest.mark.parametrize("optimize", ["0", "1"])
def test_verify_exact_pair_refuses_cross_profile_kernel(stamped_pairs, optimize):
    kernel, adapter = stamped_pairs["torch2110"]
    env = {**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
           "PYTHONOPTIMIZE": optimize}
    for actual_kernel, succeeds in [(kernel, True), (stamped_pairs["torch2130"][0], False)]:
        result = subprocess.run(["bash", "packaging/verify.sh", "--check-pair",
                                 str(actual_kernel), str(adapter)], cwd=REPO_ROOT,
                                env=env, capture_output=True, text=True)
        assert (result.returncode == 0) is succeeds, result.stdout + result.stderr


def test_verify_development_pair_accepts_public_version_pin(kernel_wheel, adapter_wheel):
    result = subprocess.run(["bash", "packaging/verify.sh", "--check-pair",
                             str(kernel_wheel), str(adapter_wheel)], cwd=REPO_ROOT,
                            env={**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stamp", [" ", "."])
def test_empty_normalized_cli_stamp_rejects_before_build(tmp_path, stamp):
    source = _copy_repo(tmp_path)
    result = subprocess.run(["bash", "packaging/build.sh"], cwd=source,
                            env={**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
                                 "PIP_NO_INDEX": "1", "DAS_BUILD_STAMP": stamp},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "empty build stamp" in result.stderr
    assert not (source / "dist").exists()


@pytest.mark.parametrize("arguments", [
    ["--runtime-profile"], ["--runtime-profile", ""],
    ["--runtime-profile", "torch2150"],
    ["--runtime-profile", "torch2110", "--runtime-profile", "torch2110"],
])
def test_cli_invalid_profile_rejects_before_build(tmp_path, arguments):
    source = _copy_repo(tmp_path)
    result = subprocess.run(["bash", "packaging/build.sh", *arguments], cwd=source,
                            env={**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
                                 "PIP_NO_INDEX": "1", "DAS_BUILD_STAMP": "2611111200.g23f605"},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "profile" in result.stderr
    assert not (source / "dist").exists()


_SENTINEL_BLOCK = """\
import importlib.util as ilu
_real = ilu.find_spec
def _blocked(name, *a, **k):
    if name == "hcu_adapter" or name.startswith("hcu_adapter."):
        return None
    return _real(name, *a, **k)
ilu.find_spec = _blocked
try:
    import sgl_kernel
except RuntimeError as e:
    assert "hcu-adapter" in str(e), e
    assert "pip install hcu-adapter" in str(e), e
    print("SENTINEL-RAISED-OK")
else:
    raise SystemExit("sgl_kernel imported without hcu_adapter: sentinel missing")
"""


def test_sgl_kernel_half_install_raises_sentinel():
    # governance §7.4 sentinel: a kernel-dist-only environment must fail
    # loudly at import instead of silently running a half-adapted kernel.
    result = subprocess.run(
        [sys.executable, "-c", _SENTINEL_BLOCK],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "HCU_ADAPTER_HOST": "none"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SENTINEL-RAISED-OK" in result.stdout
