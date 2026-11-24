#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Requires an Adapter build with HY3 launch profiles and matching HY3 runtime support.
python3 -m sglang.launch_server \
  --model-path /models/Hy3-Channel-INT8-w8a8 \
  --dp-size 8 \
  --tp-size 8 \
  --deepep-mode auto \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size=1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --watchdog-timeout 36000 \
  --cuda-graph-max-bs 16 \
  --enforce-shared-experts-fusion \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --quantization slimquant_marlin
