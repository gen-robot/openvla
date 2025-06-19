export CUDA_VISIBLE_DEVICES=7
export NCCL_SOCKET_IFNAME=bond4

NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=$((NUM_GPUS * BATCH_SIZE_PER_GPU))

# torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/train.py  \
#   --vla.type "prism-dinosiglip-224px+mx-bridge"  \
#   --vla.data_mix libero_lm_90 \
#   --pretrained_checkpoint /root/arm_ws/openvla/runs/prism-dinosiglip-224px+mx-bridge+n1+b32+x7--libero_lm_90-oxe_pretrained/checkpoints/step-015000-epoch-06-loss=0.0240.pt \
#   --is_resume True \
#   --resume_epoch 6 \
#   --resume_step 15000 \
#   --data_root_dir datasets/libero_data  \
#   --run_root_dir runs \
#   --wandb_project VLA-Reasoning \
#   --run_id_note libero_lm_90-oxe_pretrained-resumed \
#   --enable_cot True \
#   --vla.expected_world_size $NUM_GPUS \
#   --vla.global_batch_size $TOTAL_BATCH_SIZE \
#   --vla.per_device_batch_size $BATCH_SIZE_PER_GPU 

torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \
  --vla.data_mix libero_lm_90 \
  --vla.image_sequence_len 2 \
  --vla.use_wrist_image False \
  --vla.expected_world_size $NUM_GPUS \
  --vla.global_batch_size $TOTAL_BATCH_SIZE \
  --vla.per_device_batch_size $BATCH_SIZE_PER_GPU \
  --action_chunk_size 3 \
  --data_root_dir datasets/libero_data  \
  --run_root_dir runs \
  --wandb_project VLA-Reasoning \
  --run_id_note libero_lm_90-miniVLA \
  --enable_cot True
  # --cot_tags "move_reason,move"
