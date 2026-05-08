#!/bin/bash

CFG=configs/classification/cifar100/mixups/vits/convnext_tiny/convnext_t_cutmix_a2_bs100_ep200.py
GPUS=1

SEEDS=(42) # 25 7)
SPARSITIES=(0.0) # 0.4 0.5 0.6 0.7 0.8 0.9)
PORT=20000

i=0

for SEED in "${SEEDS[@]}"; do

    for SPARSITY in "${SPARSITIES[@]}"; do

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        BASELINE_ARGS=""

        if [[ "${SPARSITY}" != "0.0" ]]; then
            BASELINE_ARGS="--sparse-baseline"
        fi

        sbatch \
            --job-name=bl_s${SEED}_sp${SPARSITY} \
            tools/dist_train_convnext.sh \
            ${CFG} \
            ${GPUS} \
            ${SEED} \
            bl \
            ${SPARSITY} \
            $((PORT+i)) \
            ${BASELINE_ARGS}

        # ----------------------------------------------------
        # BLUR ONLY
        # ----------------------------------------------------

        sbatch \
            --job-name=bw_s${SEED}_sp${SPARSITY} \
            tools/dist_train_convnext.sh \
            ${CFG} \
            ${GPUS} \
            ${SEED} \
            bw \
            ${SPARSITY} \
            $((PORT+i+1)) \
            --blur --blur-depth 5

        # ----------------------------------------------------
        # SINGLE COLOR
        # ----------------------------------------------------

        sbatch \
            --job-name=sc_s${SEED}_sp${SPARSITY} \
            tools/dist_train_convnext.sh \
            ${CFG} \
            ${GPUS} \
            ${SEED} \
            sc \
            ${SPARSITY} \
            $((PORT+i+2)) \
            --blur --blur-depth 5 --single-color

        # ----------------------------------------------------
        # COLOR OPPONENCY
        # ----------------------------------------------------

        sbatch \
            --job-name=co_s${SEED}_sp${SPARSITY} \
            tools/dist_train_convnext.sh \
            ${CFG} \
            ${GPUS} \
            ${SEED} \
            co \
            ${SPARSITY} \
            $((PORT+i+3)) \
            --blur --blur-depth 5 --color-opponency

        i=$((i+4))

    done
done