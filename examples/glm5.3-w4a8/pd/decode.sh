#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export HCU_CONFIG="${HCU_CONFIG:-builtin:glm53-w4a8-decode.yaml}"

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
mkdir -p /var/log/shca
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
unset SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK

if [[ -n "${TOPO_FILE:-}" ]]; then
    if [[ ! -r "${TOPO_FILE}" ]]; then
        echo "Topology file not readable: ${TOPO_FILE}" >&2
        exit 1
    fi
    export ROCSHMEM_TOPO_FILE_FORCE="${TOPO_FILE}"
fi

PORT="${PORT:-${DECODE_PORT:-30032}}"
DIST_PORT="${DIST_PORT:-5000}"
DIST_INIT_HOST="${DIST_INIT_HOST:?Set role coordinator IP}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${DIST_INIT_HOST}:${DIST_PORT}}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?Set node rank within this role}"
TP_SIZE="${TP_SIZE:-16}"
DP_SIZE="${DP_SIZE:-16}"
EP_SIZE="${EP_SIZE:-16}"

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --trust-remote-code \
    --host "${HOST}" \
    --port "${PORT}" \
    --dist-init-addr "${DIST_INIT_ADDR}" \
    --nnodes "${NNODES}" \
    --node-rank "${NODE_RANK}" \
    --tp-size "${TP_SIZE}" \
    --moe-dense-tp-size 1 \
    --dp-size "${DP_SIZE}" \
    --ep-size "${EP_SIZE}" \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --enable-dp-lm-head \
    --deepep-mode low_latency \
    --context-length 1048576 \
    --speculative-draft-lm-head-vp-size 8 \
    --page-size 64 \
    --dsa-prefill-backend flashmla_auto \
    --dsa-decode-backend flashmla_kv \
    --quantization compressed-tensors \
    --moe-runner-backend deep_gemm \
    --dtype bfloat16 \
    --dist-timeout 10000 \
    --watchdog-timeout 3600 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.858}" \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --cuda-graph-max-bs-decode 16 \
    --max-running-requests 128 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --json-model-override-args '{"index_share_for_mtp_iteration": false}' \
    --tokenizer-worker-num 1 \
    --enable-cache-report \
    --enable-request-time-stats-logging \
    --enable-metrics \
    --expert-balancedness-report-mode both \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-ib-device "${IB_DEVICES}"
