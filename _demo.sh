#!/bin/bash
# Set the path to save video
OUTPUT_DIR='./exps/har6k_0.9/demo'
# path to video for visualization
# VIDEO_PATH='./data/har6kcsv/har6k/PXL_20220427_021841386_121.mp4'
VIDEO_PATH='./data/har6kcsv/har6k'
# path to pretrain model
# MODEL_PATH='./exps/official/checkpoint.pth' # k400_ft_vitb_ep1600_
MODEL_PATH='./exps/har6k_0.9/pre_/checkpoint-839.pth'

echo 'remember to turn on modeling_finetune.py line 138'
python3 ./run_videomae_vis_my.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
    
tar -cf ./exps/har6k_0.9/demo.tar ${OUTPUT_DIR}
