#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output_dir  \
    --summary_dir  \
    --mode train \
    --is_training True \
    --task Resnet \
    --batch_size 16 \
    --flip True \
    --random_crop True \
    --crop_size 128 \
    --input_dir_adv /home/lthpc/workspace/zhangshudong/adve/data/adv/ \
    --input_dir_nat /home/lthpc/workspace/zhangshudong/adve/data/ori/train/ \
    --num_resblock 16 \
    --name_queue_capacity 2048 \
    --image_queue_capacity 2048 \
    --perceptual_mode VGG54 \
    --queue_thread 4 \
    --ratio 0.001 \
    --learning_rate 0.000001 \
    --decay_step 84375 \
    --decay_rate 0.1 \
    --stair False \
    --beta 0.9 \
    --max_iter 168750 \
    --save_freq 5625 \


