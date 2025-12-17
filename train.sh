DATASET='domainnet'
if [[ "$DATASET" == "officehome" ]]; then
    domains=("Art" "Clipart" "Product" "Real")
elif [[ $DATASET == "domainnet" ]]; then
    domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
elif [[ $DATAET == "pacs" ]]; then
    domains=("photo" "sketch" "art_painting" "cartoon")
else
    domains=("client_0"  "client_1" "client_2" "client_3" "client_4")
fi

# 仅使用设备 0
CUDA_DEVICE=0
TRAIN_TYPE=prompt

for domain in "${domains[@]}"
do
    echo "=================================================="
    echo "STARTING TRAINING FOR DOMAIN: $domain"
    echo "=================================================="
    
    # 添加了 --report_to "tensorboard"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py \
        --output_dir "exps_$DATASET/${TRAIN_TYPE}_d_${domain}_multiclient" \
        --evaluation_type generate \
        --num_train_epochs 50 \
        --train_batch_size 4 \
        --domain "$domain" \
        --train_type $TRAIN_TYPE \
        --dataset "$DATASET" \
        --num_shot 16 \
        --learning_rate 5e-4 \
        --mixed_precision "fp16" \
        --gradient_checkpointing \
        --report_to "tensorboard" \
        --log_every_epochs 1 
    
    echo "FINISHED DOMAIN: $domain"
done