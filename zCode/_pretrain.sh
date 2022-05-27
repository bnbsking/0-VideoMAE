#!/bin/bash
OUTPUT_DIR='/home/jovyan/data-vol-1/VideoMAE/exps/har6k_0.9/pre'
DATA_PATH='/home/jovyan/data-vol-1/VideoMAE/data/har6kcsv/all.csv'

python3 run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 3200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --lr 3e-4 \
        --num_workers 8
