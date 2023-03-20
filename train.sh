#!/bin/bash
DATA_DIR=/home/litianyi/data/EgoMotion/
CONFIG_PATH=/home/litianyi/data/EgoMotion/meta_remy.yml
# DATA_DIR=/home/litianyi/data/datasets/
# CONFIG_PATH=/home/litianyi/data/datasets/meta/meta_subject_01.yml
CUDA_VISIBLE_DEVICES=7,8 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --dataset EgoMotion \
    --exp_name train03_baseline \
    --epochs 50 \
    --lr 0.01 \
    --batch_size 32 \
    --snapshot_pref baseline_stage1 \
    --gpus 0 1 \
    --eval-freq=1 \
    --clip-gradient=30 \
    --lr_steps 20 40 \
    --dropout 0.1 \
    --optimizer Adam \
    # --resume logs/train02_baseline/baseline_stage1_model_best.pth.tar \