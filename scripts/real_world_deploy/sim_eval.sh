#!/bin/bash

# GPU 设置
export CUDA_VISIBLE_DEVICES=7

# 基础路径
BASE_DIR="/mnt/public/shiliangzhi/eval-checkpoints/sim-real-co-training/pick_to_plate_multi"

# 模型列表
MODELS=(
  "openvla-7b+panda_co_training_0.5"
  "openvla-7b+panda_co_training_0.8"
  "openvla-7b+panda_co_training_0.9"
  "openvla-7b+panda_co_training_0.95"
  "openvla-7b+panda_co_training_0.99"
)

# 公共参数
COMMON_ARGS="
  --future_action_window_size 15
  --use_parallel_decoding
  --use_l1_regression
  --num_open_loop_steps 8
  --env_id PandaPutOnPlateInScene25Simple-v1
  --obs_mode rgb+segmentation
  --num_traj 1000
"

# 循环执行
for MODEL in "${MODELS[@]}"; do
  echo "==============================="
  echo " Running model: $MODEL"
  echo "==============================="

  python simualtion_deploy.py \
    --pretrained_checkpoint "${BASE_DIR}/${MODEL}/" \
    $COMMON_ARGS

done
