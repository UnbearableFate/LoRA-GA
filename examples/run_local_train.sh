#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-examples/configs/accelerate_config_local.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-examples/configs/float_Qwen3_1-7b_cola.yaml}
DEFAULT_PYTHON="${ROOT_DIR}/.venv/bin/python"
PYTHON_PATH=${PYTHON_PATH:-${DEFAULT_PYTHON}}
HF_HOME=${HF_HOME:-${ROOT_DIR}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${ROOT_DIR}/data_cache}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

cd "${ROOT_DIR}"

export MASTER_ADDR MASTER_PORT
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG}"
export HF_HOME HF_DATASETS_CACHE

if [[ ! -x "${PYTHON_PATH}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_PATH=$(command -v python3)
    else
        PYTHON_PATH=$(command -v python)
    fi
fi

echo "Using accelerate config: ${ACCELERATE_CONFIG}"
echo "Using training config: ${TRAIN_CONFIG}"
echo "Using python: ${PYTHON_PATH}"

"${PYTHON_PATH}" -u examples/train_from_config.py "${TRAIN_CONFIG}"
