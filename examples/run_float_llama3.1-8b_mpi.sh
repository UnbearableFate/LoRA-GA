#!/bin/bash
#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$(pwd)}"

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-examples/configs/accelerate_config.yaml}
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

export MASTER_ADDR MASTER_PORT
export ACCELERATE_CONFIG_FILE="$ACCELERATE_CONFIG"

ENV_VARS=("MASTER_ADDR=${MASTER_ADDR}" "MASTER_PORT=${MASTER_PORT}" "ACCELERATE_CONFIG_FILE=${ACCELERATE_CONFIG}")
ENV_LIST=$(IFS=,; echo "${ENV_VARS[*]}")
if [[ -n "${OMPI_MCA_mca_base_env_list:-}" ]]; then
    export OMPI_MCA_mca_base_env_list="${OMPI_MCA_mca_base_env_list},${ENV_LIST}"
else
    export OMPI_MCA_mca_base_env_list="${ENV_LIST}"
fi


PYTHON_PATH="/work/xg24i002/x10041/LoRA-GA/.venv/bin/python"
TRAIN_CONFIG=${TRAIN_CONFIG:-examples/configs/float_Qwen3_1-7b_metamath.yaml}

HF_HOME="/work/xg24i002/x10041/hf_home"
HF_DATASETS_CACHE="/work/xg24i002/x10041/data"

mpirun --mca mpi_abort_print_stack 1 \
       --report-bindings \
       --bind-to core \
       -np "${WORLD_SIZE}" \
       /usr/bin/env \
           MASTER_ADDR="${MASTER_ADDR}" \
           MASTER_PORT="${MASTER_PORT}" \
           ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG}" \
       bash -c "set -euo pipefail; \
                : \"\${MASTER_ADDR:?MASTER_ADDR not set}\"; \
                : \"\${MASTER_PORT:?MASTER_PORT not set}\"; \
                : \"\${ACCELERATE_CONFIG_FILE:?ACCELERATE_CONFIG_FILE not set}\"; \
                export RANK=\$OMPI_COMM_WORLD_RANK; \
                export WORLD_SIZE=\$OMPI_COMM_WORLD_SIZE; \
                export LOCAL_RANK=\$OMPI_COMM_WORLD_LOCAL_RANK; \
                export LOCAL_WORLD_SIZE=\$OMPI_COMM_WORLD_LOCAL_SIZE; \
                export HF_HOME='${HF_HOME}'; \
                export HF_DATASETS_CACHE='${HF_DATASETS_CACHE}'; \
                echo 'Running on rank' \$RANK 'out of' \$WORLD_SIZE; \
                ${PYTHON_PATH} -u examples/train_from_config.py \"${TRAIN_CONFIG}\""
