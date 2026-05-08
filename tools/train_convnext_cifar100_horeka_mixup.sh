#!/bin/bash
#SBATCH --partition=accelerated-h100,accelerated-h200-8,accelerated-h200,accelerated
#SBATCH --account=hk-project-pai00070
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=24
#SBATCH --array=0-11
#SBATCH --job-name=convnext_cifar100_horeka
#SBATCH --error=slurm/convnext_cifar100_horeka_%x_%A_%a.err
#SBATCH --output=slurm/convnext_cifar100_horeka_%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

SEEDS=(42 25 7)
CFG=configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py
preprocessings=("none" "grayscale" "single_color" "color_opponency")

SEED=${SEEDS[$(($SLURM_ARRAY_TASK_ID / 4))]}

MASTER_PORT_BASE=29511


if (( $SLURM_ARRAY_TASK_ID % 4 == 0 )); then
    preprocessing="none"
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    MASTER_PORT=$((MASTER_PORT_BASE + SLURM_ARRAY_TASK_ID))
    echo "${MASTER_PORT} Port Running with seed ${SEED} and preprocessing ${preprocessing}"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py --work_dir ${WORK_DIR} --seed ${SEED} --launcher pytorch --sparsity-threshold 0.0 --sparsity-type percentage
elif (( $SLURM_ARRAY_TASK_ID % 4 == 1 )); then
    preprocessing="grayscale"
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    MASTER_PORT=$((MASTER_PORT_BASE + SLURM_ARRAY_TASK_ID))
    echo "${MASTER_PORT} Port Running with seed ${SEED} and preprocessing ${preprocessing}"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py --work_dir ${WORK_DIR} --seed ${SEED} --launcher pytorch --sparsity-threshold 0.0 --sparsity-type percentage --blur --blur-depth 5
elif (( $SLURM_ARRAY_TASK_ID % 4 == 2 )); then
    preprocessing="single_color"
    MASTER_PORT=$((MASTER_PORT_BASE + SLURM_ARRAY_TASK_ID))
    echo "${MASTER_PORT} Port Running with seed ${SEED} and preprocessing ${preprocessing}"
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py --work_dir ${WORK_DIR} --seed ${SEED} --launcher pytorch --sparsity-threshold 0.0 --sparsity-type percentage --blur --blur-depth 5 --single-color
else
    preprocessing="color_opponency"
    MASTER_PORT=$((MASTER_PORT_BASE + SLURM_ARRAY_TASK_ID))
    echo "${MASTER_PORT} Port Running with seed ${SEED} and preprocessing ${preprocessing}"
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py --work_dir ${WORK_DIR} --seed ${SEED} --launcher pytorch --sparsity-threshold 0.0 --sparsity-type percentage --blur --blur-depth 5 --color-opponency
fi








end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime