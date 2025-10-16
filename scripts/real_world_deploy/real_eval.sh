#!/bin/bash

# GPU 设置
export CUDA_VISIBLE_DEVICES=7

CKPT_PATH="/mnt/public/chenyinuo001/openvla/checkpoints/pick_to_plate_yukun-panda_rlds_dataset_real/oft+g3tb48+openvla-7b+panda_rlds_dataset_real+b16+lr-0.0005+lora-r32+dropout-0.0+chunk-16+l1+pd/20251014_082055/openvla-7b+panda_rlds_dataset_real+b16+lr-0.0005+lora-r32+dropout-0.0+chunk-16+l1+pd--12000_chkpt"

# 公共参数
COMMON_ARGS="
  --future_action_window_size 15
  --use_parallel_decoding
  --use_l1_regression
  --num_open_loop_steps 4
"

  python franka_deploy.py \
    --pretrained_checkpoint "${CKPT_PATH}" \
    $COMMON_ARGS


