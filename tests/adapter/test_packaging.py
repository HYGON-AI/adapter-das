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
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

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


def test_stamped_wheel_pair_pins_exact_matching_build(tmp_path):
    source = _copy_repo(tmp_path)
    stamp = "20261111.g123abc"
    kernel = _build_wheel(source, tmp_path, "stamped-kernel", stamp=stamp)
    adapter = _build_wheel(source / "packaging/hcu-adapter", tmp_path,
                           "stamped-adapter", stamp=stamp)
    messages = []
    for wheel in (kernel, adapter):
        with zipfile.ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            messages.append(email.message_from_string(archive.read(name).decode()))
    expected_kernel = f"{Version(_kernel_version()).public}+hcu.{stamp}"
    expected_adapter = f"{Version(_adapter_version()).public}+hcu.{stamp}"
    assert messages[0]["Version"] == expected_kernel
    assert messages[1]["Version"] == expected_adapter
    pin = next(Requirement(r) for r in messages[1].get_all("Requires-Dist", [])
               if Requirement(r).name == "sgl-kernel")
    assert str(pin.specifier) == f"=={expected_kernel}"
    assert pin.specifier.contains(expected_kernel)
    assert not pin.specifier.contains(f"{Version(_kernel_version()).public}+hcu.other")



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


def _metadata_fixture_wheel(folder, name, version, requires=()):
    """Resolver-only distributions: no Torch/provider code is installed or run."""
    normalized = name.replace("-", "_")
    info = f"{normalized}-{version}.dist-info"
    metadata = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                + "".join(f"Requires-Dist: {item}\n" for item in requires)
                + "\nMetadata-only resolver fixture; no native ABI evidence.\n")
    payload = {
        f"{info}/METADATA": metadata.encode(),
        f"{info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for path, data in payload.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow([path, "sha256=" + digest.decode(), len(data)])
    writer.writerow([f"{info}/RECORD", "", ""])
    payload[f"{info}/RECORD"] = rows.getvalue().encode()
    wheel = folder / f"{normalized}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, data in payload.items():
            archive.writestr(path, data)
    return wheel


@pytest.mark.parametrize("torch_version,triton_version", [
    ("2.11.0", "3.6.0"), ("2.13.0", "3.7.0"),
])
@pytest.mark.parametrize("conflicting_framework_pin", [False, True])
def test_metadata_resolver_preserves_selected_framework_stack(
    kernel_wheel, adapter_wheel, tmp_path, torch_version, triton_version,
    conflicting_framework_pin,
):
    """Adapter must allow both stacks and preserve the framework's ABI constraint."""
    fixtures = tmp_path / "metadata-fixtures"
    fixtures.mkdir()
    _metadata_fixture_wheel(fixtures, "packaging", "25.0")
    _metadata_fixture_wheel(fixtures, "torch", torch_version,
                            ["triton==" + triton_version])
    for version in ["3.6.0", "3.7.0"]:
        _metadata_fixture_wheel(fixtures, "triton", version)
    requested_triton = triton_version
    if conflicting_framework_pin:
        requested_triton = "3.7.0" if triton_version == "3.6.0" else "3.6.0"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
         "--no-index", "--no-cache-dir", "--find-links", str(fixtures),
         str(kernel_wheel), str(adapter_wheel), "torch==" + torch_version,
         "triton==" + requested_triton],
        cwd=tmp_path,
        env={**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHONNOUSERSITE": "1",
             "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
             "ROCR_VISIBLE_DEVICES": ""},
        capture_output=True, text=True,
    )
    if conflicting_framework_pin:
        assert result.returncode != 0 and "ResolutionImpossible" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stamp", [
    "2611111200.g13c43d.TORCH2110", "2611111200.g13c43d.RELEASE_tag-001",
])
def test_build_cli_lists_only_the_canonical_pair(tmp_path, stamp):
    source = _copy_repo(tmp_path)
    dist = source / "dist"
    dist.mkdir()
    decoys = ["sgl_kernel-0.0.0+hcu.decoy-py3-none-any.whl",
              "hcu_adapter-0.5.22+hcu.decoy-py3-none-any.whl"]
    for name in decoys:
        (dist / name).write_bytes(b"unrelated old output")
    result = subprocess.run(
        ["bash", "packaging/build.sh"], cwd=source,
        env={**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
             "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1", "DAS_BUILD_STAMP": stamp,
             "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
             "ROCR_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name, version in [("sgl_kernel", _kernel_version()), ("hcu_adapter", _adapter_version())]:
        expected = str(Version(version.split("+", 1)[0] + "+hcu." + stamp))
        wheel = dist / f"{name}-{expected}-py3-none-any.whl"
        assert wheel.is_file()
        assert _wheel_metadata(wheel)["Version"] == expected
        assert wheel.name in result.stdout
    assert all(name not in result.stdout for name in decoys)


@pytest.fixture(scope="module")
def stamped_pairs(tmp_path_factory):
    pairs = {}
    for label, suffix in [("ordinary", ""), ("torch-text", ".TORCH2110")]:
        temp = tmp_path_factory.mktemp(label)
        source = _copy_repo(temp)
        stamp = "2611111200.g23f605" + suffix
        pairs[label] = (
            _build_wheel(source, temp, "kernel", stamp),
            _build_wheel(source / "packaging/hcu-adapter", temp, "adapter", stamp),
        )
    return pairs


def test_shared_wheel_dependencies_allow_both_existing_stack_lines(stamped_pairs):
    for kernel, adapter in stamped_pairs.values():
        km, am = _wheel_metadata(kernel), _wheel_metadata(adapter)
        adapter_base = {r.name: r for s in am.get_all("Requires-Dist", [])
                        if (r := Requirement(s)).marker is None}
        assert set(adapter_base) == {"packaging", "sgl-kernel"}
        assert str(adapter_base["packaging"].specifier) == ">=23"
        kernel_base = {r.name: r for s in km.get_all("Requires-Dist", [])
                       if (r := Requirement(s)).marker is None}
        assert str(kernel_base["torch"].specifier) == ">=2.0"
        assert str(kernel_base["triton"].specifier) == ">=2.1"
        for name, versions in [("torch", ["2.11.0", "2.13.0"]),
                               ("triton", ["3.6.0", "3.7.0"])]:
            requirements = [r for base in [adapter_base, kernel_base]
                            for key, r in base.items() if key == name]
            assert all(r.specifier.contains(version) for r in requirements for version in versions)


def test_real_wheel_stamp_does_not_select_dependency_vector(stamped_pairs):
    vectors = []
    for kernel, adapter in stamped_pairs.values():
        metadata = _wheel_metadata(adapter)
        vectors.append([s for s in metadata.get_all("Requires-Dist", [])
                        if Requirement(s).name != "sgl-kernel"])
    assert vectors[0] == vectors[1]


@pytest.mark.parametrize("label", ["ordinary", "torch-text"])
def test_real_stamped_pair_preserves_record_payload_and_interlock(stamped_pairs, label):
    kernel, adapter = stamped_pairs[label]
    km, am = _wheel_metadata(kernel), _wheel_metadata(adapter)
    base = {r.name: r for text in am.get_all("Requires-Dist", [])
            if (r := Requirement(text)).marker is None}
    terms = list(base["sgl-kernel"].specifier)
    assert len(terms) == 1 and terms[0].operator == "=="
    assert Version(terms[0].version) == Version(km["Version"])
    assert am["Version"].split("+", 1)[1] == km["Version"].split("+", 1)[1]
    for wheel in [kernel, adapter]:
        with zipfile.ZipFile(wheel) as archive:
            record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
            for name, digest, size in csv.reader(io.StringIO(archive.read(record).decode())):
                if digest:
                    data = archive.read(name)
                    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
                    assert digest == "sha256=" + encoded.decode()
                    assert int(size) == len(data)
    # Provenance must not change runtime source, extras or entry points.
    with zipfile.ZipFile(adapter) as archive:
        for name in archive.namelist():
            if name.startswith("hcu_adapter/") or name == "sitecustomize.py":
                assert archive.read(name) == (REPO_ROOT / name).read_bytes()
    assert set(am.get_all("Provides-Extra", [])) == set(
        _wheel_metadata(stamped_pairs["ordinary"][1]).get_all("Provides-Extra", []))


@pytest.mark.parametrize("stamp", [
    "2611111200.g13c43d.torch2150", "2611111200.torch2110.g13c43d",
    "2611111200.g13c43d.torch2110.torch2110",
    "2611111200.g13c43d.torch2130.torch2110",
    "2611111200.g13c43d.TORCH2110", "2611111200.g13c43d-torch2110",
    "2611111200.g13c43d_torch2110",
])
def test_backend_stamp_text_is_only_provenance(tmp_path, stamp):
    source = _copy_repo(tmp_path)
    output = tmp_path / "metadata"
    output.mkdir()
    env = {**os.environ, "HCU_ADAPTER_HOST": "none", "DAS_BUILD_STAMP": stamp,
           "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from setuptools import build_meta; "
        "build_meta.prepare_metadata_for_build_wheel(sys.argv[1])", str(output)],
        cwd=source / "packaging/hcu-adapter", env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = email.message_from_bytes(next(output.glob("*.dist-info/METADATA")).read_bytes())
    base = [s for s in metadata.get_all("Requires-Dist", [])
            if Requirement(s).marker is None and Requirement(s).name != "sgl-kernel"]
    assert base == ["packaging>=23"]


@pytest.mark.parametrize("unstamped", [False, True])
def test_build_cli_keeps_stamp_independent_of_dependencies(tmp_path, unstamped):
    source = _copy_repo(tmp_path)
    env = {**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
           "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1", "HIP_VISIBLE_DEVICES": "",
           "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
           "DAS_BUILD_STAMP": "2611111200.g13c43d.torch2110"}
    args = ["--no-stamp"] if unstamped else []
    result = subprocess.run(["bash", "packaging/build.sh", *args],
                            cwd=source, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    adapter = next((source / "dist").glob("hcu_adapter-*.whl"))
    metadata = _wheel_metadata(adapter)
    assert metadata["Version"] == (_adapter_version() if unstamped else
                                     "0.5.22+hcu.2611111200.g13c43d.torch2110")
    base = [s for s in metadata.get_all("Requires-Dist", [])
            if Requirement(s).marker is None and Requirement(s).name != "sgl-kernel"]
    assert base == ["packaging>=23"]


@pytest.mark.parametrize("optimize", ["0", "1"])
def test_verify_exact_pair_refuses_wrong_stamp_kernel(stamped_pairs, optimize):
    kernel, adapter = stamped_pairs["torch-text"]
    env = {**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
           "PYTHONOPTIMIZE": optimize}
    for actual_kernel, succeeds in [(kernel, True), (stamped_pairs["ordinary"][0], False)]:
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


@pytest.mark.parametrize("value", ["torch2110", "torch2130"])
def test_removed_runtime_profile_option_rejects_before_build(tmp_path, value):
    source = _copy_repo(tmp_path)
    result = subprocess.run(["bash", "packaging/build.sh", "--runtime-profile", value], cwd=source,
                            env={**os.environ, "HCU_ADAPTER_HOST": "none", "PYTHON": sys.executable,
                                 "PIP_NO_INDEX": "1", "DAS_BUILD_STAMP": "2611111200.g13c43d"},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "unknown argument" in result.stderr
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
