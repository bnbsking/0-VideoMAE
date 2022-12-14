#!/bin/bash
DATA_PATH='/home/jovyan/data-vol-1/VideoMAE/_data/csvDownstream/20220826_v1' # labeled
#DATA_PATH='/home/jovyan/data-vol-1/VideoMAE/_data/csvPretext/20220826_v1' # unlabeled

OUTPUT_DIR='/home/jovyan/data-vol-1/VideoMAE/_exps/downstream_0826_v1'

MODEL_PATH='/home/jovyan/data-vol-1/VideoMAE/_exps/pretext_0826_v1/checkpoint-3199.pth' # finetune
#MODEL_PATH='/home/jovyan/data-vol-1/VideoMAE/_exps/downstream_0826_v1/checkpoint-best.pth' # eval

python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set UCF101 \
    --nb_classes 4 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
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
    #--eval
    #--enable_deepspeed
