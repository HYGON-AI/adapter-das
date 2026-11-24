#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Requires installed HY3 launch profiles and matching Adapter/Plugin runtime support.
# Provide the amdsmi library directory through the deployment library path if needed.
python3 -m sglang.launch_server \
  --model-path /models/Hy3-Channel-INT8-w8a8 \
  --tp-size 8 \
  --trust-remote-code \
  --enable-cache-report \
  --mem-fraction-static 0.80 \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e5m2 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan
