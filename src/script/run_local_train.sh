#!/bin/bash

set -euo pipefail

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-src/configs/accelerate_config_local.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-src/configs/qwen3_1-7b_cola.yaml}

WORKSPACE_DIR="/work/xg24i002/x10041"
PROJECT_DIR="${WORKSPACE_DIR}/LoRA-GA"
DEFAULT_PYTHON="${PROJECT_DIR}/.venv/bin/python"
PYTHON_PATH=${PYTHON_PATH:-${DEFAULT_PYTHON}}
HF_HOME=${HF_HOME:-${WORKSPACE_DIR}/hf_home}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${WORKSPACE_DIR}/data}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

cd "${PROJECT_DIR}"

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

"${PYTHON_PATH}" src/train.py --config "${TRAIN_CONFIG}"
