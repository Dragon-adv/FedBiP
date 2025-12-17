#!/bin/bash
set -euo pipefail

# Single-GPU Windows-friendly (Git Bash): force to GPU 0 and run serially.
export CUDA_VISIBLE_DEVICES=0

DATASET="domainnet"
if [[ "$DATASET" == "officehome" ]]; then
    domains=("Art" "Clipart" "Product" "Real")
elif [[ "$DATASET" == "domainnet" ]]; then
    domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
elif [[ "$DATASET" == "pacs" ]]; then
    domains=("photo" "sketch" "art_painting" "cartoon")
else
    domains=("client_0" "client_1" "client_2" "client_3" "client_4")
fi

TRAIN_TYPE="prompt"
EXPS_ROOT="exps_${DATASET}"

# We'll generate images into a shared folder, and copy per-domain artifacts into it before each run.
GEN_OUTPUT_DIR="${EXPS_ROOT}/generated"
mkdir -p "${GEN_OUTPUT_DIR}"

TEST_TYPE="syn_wnoise_0.1_interpolated"

for domain in "${domains[@]}"; do
    echo "Generating domain: $domain"

    TRAIN_OUTPUT_DIR="${EXPS_ROOT}/${TRAIN_TYPE}_d_${domain}_multiclient"
    if [[ ! -d "${TRAIN_OUTPUT_DIR}" ]]; then
        echo "Missing train output dir: ${TRAIN_OUTPUT_DIR}"
        echo "Please run train.sh first."
        exit 1
    fi

    # Copy required artifacts into GEN_OUTPUT_DIR (generate.py loads from args.output_dir).
    shopt -s nullglob
    artifacts=(
        "${TRAIN_OUTPUT_DIR}"/prompt_domain_*.pth
        "${TRAIN_OUTPUT_DIR}"/prompt_class_*.pth
        "${TRAIN_OUTPUT_DIR}"/mean_*.pt
        "${TRAIN_OUTPUT_DIR}"/std_*.pt
        "${TRAIN_OUTPUT_DIR}"/prompt_domain.pth
        "${TRAIN_OUTPUT_DIR}"/prompt_class.pth
        "${TRAIN_OUTPUT_DIR}"/mean.pt
        "${TRAIN_OUTPUT_DIR}"/std.pt
    )
    shopt -u nullglob

    if [[ ${#artifacts[@]} -eq 0 ]]; then
        echo "No artifacts found under: ${TRAIN_OUTPUT_DIR}"
        echo "Expected prompt_domain_{idx}.pth / prompt_class_{idx}.pth / mean_{idx}.pt / std_{idx}.pt"
        exit 1
    fi

    cp -f "${TRAIN_OUTPUT_DIR}"/prompt_domain*.pth "${GEN_OUTPUT_DIR}/" 2>/dev/null || true
    cp -f "${TRAIN_OUTPUT_DIR}"/prompt_class*.pth "${GEN_OUTPUT_DIR}/" 2>/dev/null || true
    cp -f "${TRAIN_OUTPUT_DIR}"/mean*.pt "${GEN_OUTPUT_DIR}/" 2>/dev/null || true
    cp -f "${TRAIN_OUTPUT_DIR}"/std*.pt "${GEN_OUTPUT_DIR}/" 2>/dev/null || true

    python generate.py \
        --output_dir "${GEN_OUTPUT_DIR}" \
        --evaluation_type generate_prompt \
        --domain "$domain" \
        --test_type "${TEST_TYPE}" \
        --mixed_precision bf16 \
        --dataset "$DATASET"
done