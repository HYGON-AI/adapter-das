#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export HCU_CONFIG="${HCU_CONFIG:-builtin:glm53-w4a8-prefill.yaml}"

# All inference requests use local or private worker addresses.
export NO_PROXY="*" no_proxy="*"
export PYTHONDONTWRITEBYTECODE=1

# DTK26.10 candidate; set node and coordinator variables per topology.json.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NODE_HOST="${NODE_HOST:?Set node IP}"
NODE_IFNAME="${NODE_IFNAME:-bond1}"
HOST="${HOST:-${NODE_HOST}}"
SOCKET_IFNAME="${SOCKET_IFNAME:-${NODE_IFNAME}}"
IB_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"
# shellcheck source=examples/glm5.3-w4a8/pd/common.sh
source "${SCRIPT_DIR}/common.sh"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"

# Keep deployment-specific IB selection for RCCL, DeepEP, and Mooncake.
export NCCL_IB_HCA="${IB_DEVICES}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
export MC_TE_FILTERS="${IB_DEVICES}"

# Accuracy must use real speculation and expert routing.
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
unset SGLANG_SIMULATE_UNIFORM_EXPERTS SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS
unset SGLANG_SIMULATED_EXPERT_BALANCE

PORT="${PORT:-${PREFILL_PORT:-30000}}"
DIST_PORT="${DIST_PORT:-5001}"
DIST_INIT_HOST="${DIST_INIT_HOST:?Set role coordinator IP}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${DIST_INIT_HOST}:${DIST_PORT}}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?Set node rank within this role}"
PP_SIZE="${PP_SIZE:-2}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-8}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8199}"

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --pp-size "${PP_SIZE}" \
    --disable-overlap-schedule \
    --tp-size "${TP_SIZE}" \
    --dp-size "${DP_SIZE}" \
    --ep-size "${EP_SIZE}" \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --attn-cp-size "${CP_SIZE}" \
    --enable-prefill-cp \
    --cp-strategy interleave \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --context-length 1048576 \
    --kv-cache-dtype fp8_e4m3 \
    --dtype bfloat16 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.8}" \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --max-running-requests 96 \
    --disable-cuda-graph \
    --page-size 64 \
    --dsa-prefill-backend flashmla_sparse \
    --dsa-decode-backend flashmla_kv \
    --quantization compressed-tensors \
    --moe-runner-backend deep_gemm \
    --enable-cache-report \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-ib-device "${IB_DEVICES}" \
    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    --enable-dsa-cache-layer-split \
    --deepep-config "${SCRIPT_DIR}/ep_config.json"
