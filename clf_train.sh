#!/bin/bash
set -euo pipefail

# Single-GPU Windows-friendly (Git Bash): force to GPU 0 and run serially.
export CUDA_VISIBLE_DEVICES=0

NUM_SHOTS=(80)
DATASETS=("domainnet")

for DATASET in "${DATASETS[@]}"; do
    EXPS_ROOT="exps_${DATASET}"
    GEN_OUTPUT_DIR="${EXPS_ROOT}/generated"
    TEST_TYPE="syn_wnoise_0.1_interpolated"

    mkdir -p "logs_${DATASET}"

    for N_SHOT in "${NUM_SHOTS[@]}"; do
        TRAIN_TYPE="train_${TEST_TYPE}_${N_SHOT}"
        echo "Classifier training: dataset=${DATASET}, train_type=${TRAIN_TYPE}"
        python clf_train.py \
            --output_dir "${GEN_OUTPUT_DIR}" \
            --train_type "${TRAIN_TYPE}" \
            --test_type "${TEST_TYPE}" \
            --dataset "${DATASET}" \
            --pretrained \
            > "logs_${DATASET}/log_${TRAIN_TYPE}.txt"
    done
done
