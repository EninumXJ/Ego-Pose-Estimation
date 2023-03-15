#!/bin/bash
DATA_DIR=/home/litianyi/data/EgoMotion/
CONFIG_PATH=/home/litianyi/data/EgoMotion/meta_remy.yml
CUDA_VISIBLE_DEVICES=0,2,4,6 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --dataset EgoMotion \
    --exp_name train01_baseline \
    --epochs 50 \
    --lr 0.01 \
    --batch_size 48 \
    --snapshot_pref baseline_stage1 \
    --gpus 0 1 2 3 \
    --eval-freq=1 \
    --clip-gradient=50 \
    --lr_steps 20 40 \
    --dropout 0.5 \
    --optimizer Adam \
    # --resume logs/train16/baseline_stage1_model_best.pth.tar \