"""Build-time version pin shared by setup.py and ci/l0_gate.py.

DAS_BUILD_STAMP (<time>.g<commit>) is set by packaging/build.sh and rendered
as the local version hcu.<time>.g<commit>, so both wheels of one build carry
the same local label and hcu-adapter depends on the exact stamped kernel
build. Without the stamp (editable/dev installs) the adapter falls back to
its VERSION file and the kernel's public version.

The stamp records provenance only; it never selects Torch/Triton dependencies.
The actual Core/provider metadata and deployment constraints own that choice.
"""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_VERSION_FILE = HERE.parent.parent / "sgl_kernel" / "version.py"


def build_stamp():
    return os.environ.get("DAS_BUILD_STAMP", "").strip().strip(".") or None


def kernel_public_version():
    src = KERNEL_VERSION_FILE.read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "(.+?)"$', src, re.M)
    if not m:
        raise RuntimeError(f"cannot read __version__ from {KERNEL_VERSION_FILE}")
    return m.group(1).split("+", 1)[0]


def kernel_pin(stamp=None):
    if stamp is None:
        stamp = build_stamp()
    version = kernel_public_version()
    return f"sgl-kernel=={version}+hcu.{stamp}" if stamp else f"sgl-kernel=={version}"


def adapter_version(stamp=None):
    if stamp is None:
        stamp = build_stamp()
    version = (HERE / "VERSION").read_text(encoding="utf-8").strip()
    return f"{version.split('+', 1)[0]}+hcu.{stamp}" if stamp else version


def base_dependencies():
    """Host-neutral requirements; runtime stack selection belongs to consumers."""
    return ["packaging>=23"]
