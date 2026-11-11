"""Build-time version pin shared by setup.py and ci/l0_gate.py.

DAS_BUILD_STAMP (<time>.g<commit>) is set by packaging/build.sh and rendered
as the local version hcu.<time>.g<commit>, so both wheels of one build carry
the same local label and hcu-adapter depends on the exact stamped kernel
build. Without the stamp (editable/dev installs) the adapter falls back to
its VERSION file and the kernel's public version.

The reserved final stamp segment ``torch2110`` selects the retained
Torch 2.11 / Triton 3.6 dependency profile. Unmarked builds keep the
Torch 2.13 / Triton 3.7 default. This is build metadata, not runtime ABI admission.
"""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_VERSION_FILE = HERE.parent.parent / "sgl_kernel" / "version.py"


def build_stamp():
    return os.environ.get("DAS_BUILD_STAMP", "").strip().strip(".") or None


def runtime_profile(stamp=None):
    if stamp is None:
        stamp = build_stamp()
    # PEP440 normalizes '-'/'_' and case in local versions. Reject aliases
    # rather than letting a default-profile wheel normalize to a 2110 name.
    parts = re.split(r"[._-]", stamp or "")
    markers = [part for part in parts if part.lower().startswith("torch")]
    if markers and (len(markers) != 1 or markers[0] != parts[-1]
                    or markers[0] not in {"torch2110", "torch2130"}
                    or not stamp.endswith("." + markers[0])):
        raise ValueError(f"invalid runtime profile in build stamp: {stamp!r}")
    return markers[0] if markers else "torch2130"


def profile_stamp(stamp, profile=None):
    """Normalize one backend stamp, refusing conflicting profile selectors."""
    stamp = stamp.strip().strip(".")
    current = runtime_profile(stamp)
    if profile is not None and profile not in {"torch2110", "torch2130"}:
        raise ValueError(f"unknown runtime profile: {profile!r}")
    marked = any(part.startswith("torch") for part in stamp.split("."))
    if marked and profile is not None and profile != current:
        raise ValueError(f"runtime profile conflict: {profile} versus stamp {stamp!r}")
    if profile == "torch2110" and not marked:
        if not stamp:
            raise ValueError("runtime profile torch2110 requires a build stamp")
        return stamp + ".torch2110"
    return stamp


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
    runtime_profile(stamp)
    version = (HERE / "VERSION").read_text(encoding="utf-8").strip()
    return f"{version.split('+', 1)[0]}+hcu.{stamp}" if stamp else version


def base_dependencies(stamp=None):
    """Validated DTK stack pins carried by the adapter dist.

    Unmarked builds mirror the release line. The explicit stamped profile
    preserves the retained 2.11/3.6 cohort without weakening either pin.
    """
    if runtime_profile(stamp) == "torch2110":
        return ["packaging>=23", "torch==2.11.0", "triton==3.6.0"]
    return ["packaging>=23", "torch==2.13.0", "triton==3.7.0"]
