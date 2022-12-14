#!/bin/bash
DATA_PATH='/home/jovyan/data-vol-1/VideoMAE/_data/csvDownstream/20220826_v1'
OUTPUT_DIR='/home/jovyan/data-vol-1/VideoMAE/_exps/downstream_0826_v1'

for FINETUNE_WEIGHT in $(ls $OUTPUT_DIR | grep checkpoint-1[0-9][0-9][0-9]); do 
    FINETUNE_WEIGHT=$OUTPUT_DIR/$FINETUNE_WEIGHT
    echo ---$FINETUNE_WEIGHT---
    echo ---$FINETUNE_WEIGHT--- >> $OUTPUT_DIR/eval.txt
    python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set UCF101 \
    --nb_classes 5 \
    --data_path ${DATA_PATH} \
    --finetune ${FINETUNE_WEIGHT} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 1e-3 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 2000 \
    --dist_eval \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --mixup 0 \
    --cutmix 0 \
    --smoothing 0 \
    --eval >> $OUTPUT_DIR/eval.txt
done