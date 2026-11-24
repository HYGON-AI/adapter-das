#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Requires installed HY3 launch profiles and matching Adapter/Plugin runtime support.
python3 -m sglang.launch_server \
  --model-path /models/Hy3-Channel-FP8-w8a8 \
  --tp-size 8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan
