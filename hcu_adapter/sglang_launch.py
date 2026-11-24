# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch model launch configuration without importing GPU operators."""


def apply_official_defaults(argv, original):
    from .hy3_launch import apply_official_defaults as apply_hy3

    if apply_hy3(argv, original):
        return
    from .glm53_launch import apply_official_defaults as apply_glm53

    apply_glm53(argv, original)
