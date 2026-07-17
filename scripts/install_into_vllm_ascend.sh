#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${VLLM_ASCEND_CHECKOUT:-${HOME}/vllm_ascend}}"

if [[ ! -d "${TARGET_DIR}/vllm_ascend/patch/platform" ]]; then
  echo "Target does not look like a vllm_ascend checkout: ${TARGET_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}/vllm_ascend/occamtoken"
cp -a "${ROOT_DIR}/src/ascend_occamtoken/." "${TARGET_DIR}/vllm_ascend/occamtoken/"

cp "${ROOT_DIR}/patches/platform/patch_occamtoken.py" \
  "${TARGET_DIR}/vllm_ascend/patch/platform/patch_occamtoken.py"
cp "${ROOT_DIR}/patches/worker/patch_occamtoken_qwen35.py" \
  "${TARGET_DIR}/vllm_ascend/patch/worker/patch_occamtoken_qwen35.py"
cp "${ROOT_DIR}/patches/worker/patch_occamtoken_stage2_decoder.py" \
  "${TARGET_DIR}/vllm_ascend/patch/worker/patch_occamtoken_stage2_decoder.py"
cp "${ROOT_DIR}/patches/worker/patch_occamtoken_runner.py" \
  "${TARGET_DIR}/vllm_ascend/patch/worker/patch_occamtoken_runner.py"

INIT_FILE="${TARGET_DIR}/vllm_ascend/patch/platform/__init__.py"
if grep -q "OccamToken experiment patch. Installed from /" "${INIT_FILE}"; then
  sed -i \
    's|# OccamToken experiment patch\. Installed from /.*|# OccamToken experiment patch. Installed from an external experiment checkout.|' \
    "${INIT_FILE}"
fi

if ! grep -q "patch_occamtoken" "${INIT_FILE}"; then
  cat >> "${INIT_FILE}" <<'PY'

# OccamToken experiment patch. Installed from an external experiment checkout.
if os.getenv("VLLM_ASCEND_OCCAMTOKEN_ENABLE", "0").lower() in ("1", "true", "yes", "on"):
    import vllm_ascend.patch.platform.patch_occamtoken  # noqa
PY
fi

echo "Installed OccamToken experiment patch into ${TARGET_DIR}"
