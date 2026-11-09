# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single source of the sgl-kernel dist version (governance §7.4).

The shadow kernel wheel reads its version from here; the hcu-adapter dist
carries an independent version line of its own (packaging/hcu-adapter/
VERSION). The interlock pin in packaging/hcu-adapter/pyproject.toml must
equal THIS version (ci/l0_gate checks it; packaging/build.sh re-verifies).
The base tracks the *kernel* version of the paired upstream sglang tree
(v0.5.22 tag ships python/sglang/kernels/aot at 0.4.9, and its engine.py
asserts sglang-kernel >= 0.4.9), NOT the sglang release number. Re-pair
with an upstream tree => bump the base to that tree's aot pyproject
version.
"""

__version__ = "0.4.9+hcu.1"
