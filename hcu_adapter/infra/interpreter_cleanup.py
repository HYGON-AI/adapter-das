# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free interpreter cleanup for frozen cyclic native owners."""

import atexit
import gc

_registered = False


def register_frozen_heap_cleanup() -> None:
    """Restore cyclic finalization at exit, retaining the serving GC policy.

    Host startup serializes installation. The callback and registration flag
    are inherited together by fork children; spawn initializes its own copy.
    Capture the builtin directly so cleanup needs no module globals at exit.
    """
    global _registered
    if not _registered:
        atexit.register(gc.unfreeze)
        _registered = True
