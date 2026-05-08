#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=05:59:59
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lorena.stracke@uni-siegen.de
#SBATCH --export=ALL
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e

PYTHON=${PYTHON:-python}

CFG=$1
GPUS=$2
SEED=$3
PREPROC=$4
SPARSITY=$5
PORT=$6

shift 6
EXTRA_ARGS="$@"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
#${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# ------------------------------------------------------------
# preprocessing short names
# ------------------------------------------------------------

# bl = baseline
# bw = blur only
# sc = single color
# co = color opponency

SPARSITY_STR=$(printf "%.1f" ${SPARSITY} | tr '.' 'p')

WORK_DIR="work_dir_convnext_tiny_preprocessing_seed${SEED}_${PREPROC}_sparse_percentage${SPARSITY_STR}"

mkdir -p logs
mkdir -p ${WORK_DIR}


echo "=================================================="
echo "CFG        : ${CFG}"
echo "SEED       : ${SEED}"
echo "PREPROC    : ${PREPROC}"
echo "SPARSITY   : ${SPARSITY}"
echo "PORT       : ${PORT}"
echo "WORK_DIR   : ${WORK_DIR}"
echo "EXTRA_ARGS : ${EXTRA_ARGS}"
echo "=================================================="

$PYTHON -m torch.distributed.launch \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    tools/train.py ${CFG} \
    --work-dir ${WORK_DIR} \
    --seed ${SEED} \
    --launcher pytorch \
    --sparsity-threshold ${SPARSITY} \
    --sparsity-type percentage \
    ${EXTRA_ARGS} 