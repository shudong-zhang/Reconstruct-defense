# !/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --output_dir your path\
    --summary_dir your path \
    --mode inference \
    --is_training False \
    --task Resnet \
    --batch_size 16 \
    --input_dir_LR your path\
    --input_dir_HR your path \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint your path
