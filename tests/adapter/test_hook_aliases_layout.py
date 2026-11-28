# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The Hook alias optimizer must load through main's hcu_adapter.infra layout.

Release PR #206 wrote ``hook_aliases`` against the pre-refactor layout
(``from hcu_adapter import hcu_env, registry``). The sglang bridge now
guards the call, so a failure there only costs this bridge; the import of
the layout is still what this case pins. No sglang import is needed for
the ``0`` case: install() returns before touching sglang, so it runs on
every host. The unset case is the default-on path since 2026-11-28: it
needs the upstream module and skips without it; on a host without the
native helper it retains stock, on a build-capable host it binds, so only
the return type is portable.
"""

import pytest

from hcu_adapter import hook_aliases


@pytest.mark.parametrize("switch", [None, "0"], ids=["unset", "0"])
def test_install_loads_through_the_infra_layout(monkeypatch, switch):
    monkeypatch.delenv("HCU_HOOK_ALIAS_FAST", raising=False)
    if switch is not None:
        monkeypatch.setenv("HCU_HOOK_ALIAS_FAST", switch)
        assert hook_aliases.install() is False
        return
    # the unset case is the default-on path: it needs the upstream module
    hook_registry = pytest.importorskip("sglang.srt.plugins.hook_registry")
    # a build-capable host binds the native scan: put the upstream helper back afterwards
    monkeypatch.setattr(hook_registry, "_propagate_patch", hook_registry._propagate_patch)
    assert isinstance(hook_aliases.install(), bool)
