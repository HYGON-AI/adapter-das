#!/usr/bin/env bash
# Release-flow install-combo verification over the two wheels built by
# packaging/build.sh. Three combos (dual-dist semantics, governance §7.4):
#   A sglang combo   -- both wheels: shadow import, entry-points, sitecustomize
#   B vLLM-only      -- hcu-adapter alone: entry-points + sitecustomize live
#                        with it; sgl_kernel is not installed in the venv
#   C half-install   -- sgl-kernel alone: import sgl_kernel must raise the
#                        sentinel (unless an ambient sgl_kernel shadows it)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3 python; do
        if "$cand" -c "import sys" >/dev/null 2>&1; then PY="$cand"; break; fi
    done
fi
[ -n "$PY" ] || { echo "verify.sh: no working python3/python on PATH" >&2; exit 1; }

# Mixed build directories require an explicit pair. --check-pair only
# inspects real wheel metadata; it creates no environment and installs nothing.
CHECK_PAIR=0
if [ "${1:-}" = --check-pair ]; then CHECK_PAIR=1; shift; fi
if [ "$#" = 2 ]; then
    KERNEL_WHL="$1"; ADAPTER_WHL="$2"
elif [ "$#" = 0 ] && [ "$CHECK_PAIR" = 0 ]; then
    shopt -s nullglob
    kernels=("$DIST"/sgl_kernel-*.whl)
    adapters=("$DIST"/hcu_adapter-*.whl)
    if [ "${#kernels[@]}" != 1 ] || [ "${#adapters[@]}" != 1 ]; then
        echo "verify.sh: specify exact kernel and adapter wheel paths for an ambiguous dist directory" >&2
        exit 2
    fi
    KERNEL_WHL="${kernels[0]}"; ADAPTER_WHL="${adapters[0]}"
else
    echo "usage: verify.sh [--check-pair] <kernel.whl> <adapter.whl>" >&2; exit 2
fi
"$PY" - "$KERNEL_WHL" "$ADAPTER_WHL" <<'PY'
import email
import sys
import zipfile
from packaging.requirements import Requirement
def metadata(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        return email.message_from_bytes(archive.read(name))
kernel, adapter = map(metadata, sys.argv[1:])
if kernel["Name"] != "sgl-kernel" or adapter["Name"] != "hcu-adapter":
    raise SystemExit("wrong wheel owners")
pins = [Requirement(r) for r in adapter.get_all("Requires-Dist", [])
        if Requirement(r).name == "sgl-kernel"]
terms = list(pins[0].specifier) if len(pins) == 1 else []
if (len(pins) != 1 or pins[0].marker is not None or len(terms) != 1
        or terms[0].operator != "==" or "*" in terms[0].version
        or not terms[0].contains(kernel["Version"], prereleases=True)):
    raise SystemExit(f"wheel pair mismatch: {pins!r} versus {kernel['Version']}")
print("Exact wheel pair:", kernel["Version"], adapter["Version"])
PY
if [ "$CHECK_PAIR" = 1 ]; then exit 0; fi

mkvenv() {  # torch/triton come from the system site; wheels install --no-deps
    "$PY" -m venv --system-site-packages "$1"
}

venv_py() {  # linux venvs use bin/, windows venvs use Scripts/
    if [ -x "$1/bin/python" ]; then echo "$1/bin/python"
    else echo "$1/Scripts/python.exe"; fi
}

run_py() {  # run_py <venv> <code>
    "$(venv_py "$1")" -c "$2"
}

venv_pip() {  # venv_pip <venv> <args...>
    "$(venv_py "$1")" -m pip "${@:2}"
}

echo "== combo A: sglang (both wheels) =="
VA="$(mktemp -d)/va"; mkvenv "$VA"
venv_pip "$VA" install --no-deps --quiet "$KERNEL_WHL" "$ADAPTER_WHL"
run_py "$VA" '
import sys, importlib.metadata as im
from packaging.requirements import Requirement
import sgl_kernel
assert "sitecustomize" in sys.modules, "sitecustomize not effective"
eps = {e.name: e for e in im.entry_points(group="vllm.platform_plugins")
       if (e.dist.name if e.dist else "") == "hcu-adapter"}
assert "hcu_adapter" in eps, "vllm platform plugin entry-point missing"
pin = [r.replace(" ", "") for r in (im.metadata("hcu-adapter").get_all("Requires-Dist") or [])
       if r.startswith("sgl-kernel")]
assert len(pin) == 1 and Requirement(pin[0]).specifier.contains(
    im.version("sgl-kernel"), prereleases=True), (pin, im.version("sgl-kernel"))
print("A ok:", sgl_kernel.__version__)
'

echo "== combo B: vLLM-only (hcu-adapter alone) =="
VB="$(mktemp -d)/vb"; mkvenv "$VB"
venv_pip "$VB" install --no-deps --quiet "$ADAPTER_WHL"
if venv_pip "$VB" show sgl-kernel >/dev/null 2>&1; then
    echo "B FAIL: sgl-kernel must not be installed in this venv" >&2; exit 1
fi
run_py "$VB" '
import sys, importlib.metadata as im
assert "sitecustomize" in sys.modules
eps = [e for e in im.entry_points(group="vllm.general_plugins")
       if (e.dist.name if e.dist else "") == "hcu-adapter"]
assert eps, "vllm general plugin entry-point missing"
try:
    import sgl_kernel
except RuntimeError:
    pass
else:
    print("B note: ambient sgl_kernel shadowed the sentinel path; "
          "unit test test_sgl_kernel_*_sentinel covers the clean case")
print("B ok")
'

echo "== combo C: half-install (sgl-kernel alone) =="
VC="$(mktemp -d)/vc"; mkvenv "$VC"
venv_pip "$VC" install --no-deps --quiet "$KERNEL_WHL"
run_py "$VC" '
import importlib.util as ilu
import sys
try:
    import sgl_kernel
except RuntimeError as e:
    assert "hcu-adapter" in str(e), e
    print("C ok: sentinel message:", str(e)[:70])
else:
    # an ambient hcu_adapter (e.g. an editable install leaking through
    # --system-site-packages) shadowed the half-install; re-probe with
    # hcu_adapter blocked so the sentinel branch is deterministic
    real = ilu.find_spec
    def blocked(name, *a, **k):
        if name == "hcu_adapter" or name.startswith("hcu_adapter."):
            return None
        return real(name, *a, **k)
    ilu.find_spec = blocked
    for m in [m for m in sys.modules if m == "sgl_kernel" or m.startswith("sgl_kernel.")]:
        del sys.modules[m]
    try:
        import sgl_kernel
    except RuntimeError as e:
        assert "hcu-adapter" in str(e), e
        print("C ok (ambient shadowed; blocked re-probe):", str(e)[:70])
    else:
        raise SystemExit("C FAIL: sgl_kernel imported without hcu-adapter")
'

echo "== verify.sh: all combos passed =="
