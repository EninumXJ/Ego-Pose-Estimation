#!/bin/bash
DATA_DIR=/home/liumin/litianyi/workspace/data/datasets/
CONFIG_PATH=/home/liumin/litianyi/workspace/data/datasets/meta/meta_subject_05.yml
CUDA_VISIBLE_DEVICES=1,2,3 python train.py \
    --dataset_path $DATA_DIR \
    --config_path $CONFIG_PATH \
    --exp_name train16 \
    --epochs 150 \
    --lr 0.01 \
    --batch_size 48 \
    --snapshot_pref baseline_stage1 \
    --gpus 0 1 \
    --eval-freq=5 \
    --clip-gradient=50 \
    --lr_steps 30 60 \
    --dropout 0.5 \
    --optimizer Adam \
    --resume logs/train16/baseline_stage1_model_best.pth.tar \