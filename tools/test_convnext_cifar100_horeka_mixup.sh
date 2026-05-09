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
#SBATCH --job-name=test_mixup_convnext_cifar100_horeka
#SBATCH --error=slurm/test_convnext_cifar100_horeka_%x_%A_%a.err
#SBATCH --output=slurm/test_convnext_cifar100_horeka_%x_%A_%a.out
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
    CKPT_PATH=$(ls ${WORK_DIR}/best_head0_top1_epoch_*.pth | tail -n 1)


    SPARSITY_THRESHOLD=0.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=10.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=20.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=30.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=40.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=50.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=60.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=70.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=80.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions

    SPARSITY_THRESHOLD=90.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --eval-corruptions


elif (( $SLURM_ARRAY_TASK_ID % 4 == 1 )); then
    preprocessing="grayscale"
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    CKPT_PATH=$(ls ${WORK_DIR}/best_head0_top1_epoch_*.pth | tail -n 1)

    SPARSITY_THRESHOLD=0.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions


    SPARSITY_THRESHOLD=10.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=20.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=30.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=40.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=50.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=60.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=70.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=80.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions

    SPARSITY_THRESHOLD=90.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --eval-corruptions


elif (( $SLURM_ARRAY_TASK_ID % 4 == 2 )); then
    preprocessing="single_color"    
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    CKPT_PATH=$(ls ${WORK_DIR}/best_head0_top1_epoch_*.pth | tail -n 1)


    SPARSITY_THRESHOLD=0.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=10.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=20.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=30.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=40.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=50.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=60.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=70.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=80.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions

    SPARSITY_THRESHOLD=90.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --single-color --eval-corruptions
else
    preprocessing="color_opponency"   
    WORK_DIR=experiments/cifar100/convnext_tiny/${preprocessing}/mixup/seed_${SEED}
    CKPT_PATH=$(ls ${WORK_DIR}/best_head0_top1_epoch_*.pth | tail -n 1)

    SPARSITY_THRESHOLD=0.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=10.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=20.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=30.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=40.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=50.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=60.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=70.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=80.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions

    SPARSITY_THRESHOLD=90.0
    SAVE_DIR=${WORK_DIR}/${SPARSITY_THRESHOLD}
    CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py $CKPT_PATH --work_dir ${SAVE_DIR} --seed ${SEED} --launcher none --sparsity-threshold $SPARSITY_THRESHOLD --sparsity-type percentage --blur --blur-depth 5 --color-opponency --eval-corruptions
fi








end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime