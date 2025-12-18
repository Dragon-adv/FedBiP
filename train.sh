#!/bin/bash
set -euo pipefail

# Single-GPU Windows-friendly (Git Bash): force to GPU 0 and run serially (no '&').
export CUDA_VISIBLE_DEVICES=0

DATASET="domainnet"
if [[ "$DATASET" == "officehome" ]]; then
    domains=("Art" "Clipart" "Product" "Real")
elif [[ "$DATASET" == "domainnet" ]]; then
    domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
elif [[ "$DATASET" == "pacs" ]]; then
    domains=("photo" "sketch" "art_painting" "cartoon")
else
    # For dermamnist/ucm-like datasets.
    domains=("client_0" "client_1" "client_2" "client_3" "client_4")
fi

TRAIN_TYPE="prompt"
EXPS_ROOT="exps_${DATASET}"

for domain in "${domains[@]}"; do
    echo "Training domain: $domain"
    python train.py \
        --output_dir "${EXPS_ROOT}/${TRAIN_TYPE}_d_${domain}_multiclient" \
        --evaluation_type generate \
        --num_train_epochs 50 \
        --train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --mixed_precision bf16 \
        --domain "$domain" \
        --train_type "$TRAIN_TYPE" \
        --dataset "$DATASET" \
        --num_shot 16 \
        --learning_rate 0.1 \
        --num_workers 0
done
