#!/usr/bin/env bash
set -euo pipefail

# Fill this in for your environment before running.
: "${MODEL_PATH:?Set MODEL_PATH to your Qwen3.5 model path before running}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

run_stage() {
  local stage="$1"
  local target_ratio="$2"
  local stage1_ratio="$3"

  echo "=== stage=${stage} target_ratio=${target_ratio} stage1_ratio=${stage1_ratio} ==="
  VLLM_ASCEND_OCCAMTOKEN_ENABLE=1 \
  VLLM_ASCEND_OCCAMTOKEN_STAGE="${stage}" \
  VLLM_ASCEND_OCCAMTOKEN_TARGET_RATIO="${target_ratio}" \
  VLLM_ASCEND_OCCAMTOKEN_STAGE1_RATIO="${stage1_ratio}" \
  VLLM_ASCEND_OCCAMTOKEN_LOG_STATS=1 \
  vllm serve "${MODEL_PATH}" --host "${HOST}" --port "${PORT}"
}

case "${1:-}" in
  fixed-256)
    run_stage fixed 0.125 0.125
    ;;
  stage1-512)
    run_stage stage1 0.25 0.25
    ;;
  stage1-256)
    run_stage stage1 0.125 0.125
    ;;
  stage2-256)
    run_stage stage2 0.125 0.25
    ;;
  full-256)
    run_stage full 0.125 0.25
    ;;
  *)
    echo "Usage: $0 {fixed-256|stage1-512|stage1-256|stage2-256|full-256}" >&2
    exit 2
    ;;
esac
