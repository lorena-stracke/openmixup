#!/bin/bash
#SBATCH --partition=accelerated-h100,accelerated-h200-8,accelerated-h200,accelerated
#SBATCH --account=hk-project-pai00070
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=100gb
#SBATCH --cpus-per-task=24
#SBATCH --array=0-29
#SBATCH --job-name=test_baseline_mixup_convnext_cifar100_horeka
#SBATCH --error=slurm/test2_baseline_convnext_cifar100_horeka_%x_%A_%a.err
#SBATCH --output=slurm/test2_baseline_convnext_cifar100_horeka_%x_%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shashank.agnihotri@uni-mannheim.de

echo "Started at $(date)";
echo "Running job: $SLURM_JOB_NAME array id: $SLURM_ARRAY_TASK_ID using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

SEEDS=(42 25 7)
CFG=configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py
# preprocessings=("none" "grayscale" "single_color" "color_opponency")
preprocessing="none"
SPARSITY_THRESHOLDS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

SEED=${SEEDS[$(($SLURM_ARRAY_TASK_ID / 10))]}
SPARSITY_THRESHOLD=${SPARSITY_THRESHOLDS[$(($SLURM_ARRAY_TASK_ID % 10))]}

    
WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
CKPT_PATH=$(ls ${WORK_DIR}/best_head0_top1_epoch_*.pth | tail -n 1)


SAVE_DIR=${WORK_DIR}/sparsity_redone_${SPARSITY_THRESHOLD}
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions --sparse-baseline



end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime