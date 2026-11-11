#!/usr/bin/env bash
# Build BOTH adapter-das wheels from the single version source
# (sgl_kernel/version.py -- governance §7.4). One command, two artifacts:
#   dist/sgl_kernel-<version>+hcu.<stamp>-*.whl   (pure shadow kernel)
#   dist/hcu_adapter-<version>+hcu.<stamp>-*.whl  (adapter: bootstrap + bridges + hooks)
# Both wheels of one run carry the same hcu.<time>.g<commit> local version, and
# hcu-adapter pins the exact stamped kernel version (build-time interlock).
# --no-stamp builds the raw version lines without the time/commit local label.
# --runtime-profile torch2110 selects retained Torch 2.11 / Triton 3.6 builds;
# default torch2130 stays Torch 2.13 / Triton 3.7. Alternate builds need a stamp.
set -euo pipefail

USE_STAMP=1
RUNTIME_PROFILE=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-stamp) USE_STAMP=0; shift ;;
        --runtime-profile)
            if [ "$#" -lt 2 ] || [ -z "${2:-}" ] || [ -n "$RUNTIME_PROFILE" ]; then
                echo "build.sh: --runtime-profile needs one unique profile" >&2; exit 2
            fi
            RUNTIME_PROFILE="$2"; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) echo "build.sh: unknown argument: $1" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/sgl_kernel/version.py")"
if [ -z "$VERSION" ]; then
    echo "build.sh: cannot read __version__ from sgl_kernel/version.py" >&2
    exit 1
fi

# the adapter dist carries its own independent version line (hand-edited
# single source: packaging/hcu-adapter/VERSION)
ADAPTER_VERSION="$(tr -d ' \t\r\n' < "$ROOT/packaging/hcu-adapter/VERSION")"
if [ -z "$ADAPTER_VERSION" ]; then
    echo "build.sh: packaging/hcu-adapter/VERSION is empty" >&2
    exit 1
fi

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    # probe: some Windows boxes ship a Microsoft-Store python3 stub that
    # exists on PATH but cannot run
    for cand in python3 python; do
        if "$cand" -c "import sys" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
fi
[ -n "$PY" ] || { echo "build.sh: no working python3/python on PATH" >&2; exit 1; }

# Validate the sole backend-visible profile selector before installing/building.
RAW_STAMP="${DAS_BUILD_STAMP:-}"
if [ "$USE_STAMP" = 1 ] && [ -z "$RAW_STAMP" ]; then
    RAW_STAMP="$(date +%y%m%d%H%M).g$(git -C "$ROOT" rev-parse --short=6 HEAD)"
fi
DAS_BUILD_STAMP="$(SGL_ADAPTER_ROOT="$ROOT" "$PY" - "$RAW_STAMP" "$RUNTIME_PROFILE" "$USE_STAMP" <<'PY'
import os
import sys
sys.path.insert(0, os.path.join(os.environ["SGL_ADAPTER_ROOT"], "packaging", "hcu-adapter"))
import _pin
try:
    stamp = _pin.profile_stamp(sys.argv[1] or "development", sys.argv[2] or None)
    if sys.argv[3] == "1" and not stamp:
        raise ValueError("empty build stamp after normalization")
    if sys.argv[3] == "0":
        if _pin.runtime_profile(stamp) != "torch2130":
            raise ValueError("--no-stamp cannot use runtime profile torch2110")
        stamp = ""
    print(stamp)
except ValueError as exc:
    raise SystemExit(f"build.sh: {exc}")
PY
)"

# pip wheel --no-build-isolation needs the build backend in this environment
if ! "$PY" -c "import setuptools.build_meta" >/dev/null 2>&1; then
    echo "== installing build deps (setuptools/wheel) =="
    "$PY" -m pip install "setuptools>=64" wheel
fi

# one stamp for both wheels: <YYMMDDHHMM>.g<commit6> (rendered as hcu.<stamp>),
# alternate profile is a reserved final segment; default format stays unchanged
if [ "$USE_STAMP" = 1 ]; then
    export DAS_BUILD_STAMP
    KERNEL_BUILD_VERSION="${VERSION%%+*}+hcu.${DAS_BUILD_STAMP}"
    ADAPTER_BUILD_VERSION="${ADAPTER_VERSION%%+*}+hcu.${DAS_BUILD_STAMP}"
else
    unset DAS_BUILD_STAMP
    KERNEL_BUILD_VERSION="$VERSION"
    ADAPTER_BUILD_VERSION="$ADAPTER_VERSION"
fi

# pin check: _pin.py is the build-time interlock source, ci/l0_gate re-checks
EXPECTED_PIN="sgl-kernel==${VERSION%%+*}"
ACTUAL_PIN="$(SGL_ADAPTER_ROOT="$ROOT" "$PY" -c '
import os, sys
sys.path.insert(0, os.path.join(os.environ["SGL_ADAPTER_ROOT"], "packaging", "hcu-adapter"))
import _pin
print(_pin.kernel_pin(stamp=""))
')"
if [ "$ACTUAL_PIN" != "$EXPECTED_PIN" ]; then
    echo "build.sh: hcu-adapter interlock pin ${ACTUAL_PIN}" \
         "does not match public version ${EXPECTED_PIN#sgl-kernel==} (governance §7.4)" >&2
    exit 1
fi

mkdir -p "$ROOT/dist"

echo "== building sgl-kernel ${KERNEL_BUILD_VERSION} (repo root) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$ROOT"

echo "== building hcu-adapter ${ADAPTER_BUILD_VERSION} (packaging/hcu-adapter) =="
"$PY" -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir "$ROOT/dist" "$ROOT/packaging/hcu-adapter"

ls -l "$ROOT/dist"/sgl_kernel-"${KERNEL_BUILD_VERSION}"-*.whl "$ROOT/dist"/hcu_adapter-"${ADAPTER_BUILD_VERSION}"-*.whl
