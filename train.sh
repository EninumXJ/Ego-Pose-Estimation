#!/bin/bash
# DATA_DIR=/home/liumin/litianyi/workspace/data/datasets
# CONFIG_PATH=/home/liumin/litianyi/workspace/data/datasets/meta/meta_subject_01.yml
DATA_DIR=/home/liumin/litianyi/workspace/data/EgoMotion
CONFIG_PATH=/home/liumin/litianyi/workspace/data/EgoMotion/meta_remy.yml
CUDA_VISIBLE_DEVICES=1,2 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name train24 \
    --dataset EgoMotion \
    --epochs 30 \
    --lr 0.1 \
    --batch_size 32 \
    --snapshot_pref transformer \
    --gpus 0 1 \
    --eval-freq=1 \
    --clip-gradient=30 \
    --L 20 \
    --h 10 \
    --pose_dim 48 \
    --dff 1440 \
    --N 16 \
    --norm L1 \
    --lr_steps 50 100 \
    --optimizer Adam \
    --dropout 0.1 \
    # --resume logs/train22/transformer_checkpoint.pth.tar \
    # --no_feature