task_name="panda_simpler_sft_dataset"
CUDA_VISIBLE_DEVICES=3,5 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 4000 \
  --save_steps 100 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project RLVLA \
  --wandb_entity pancake-wbw-

#mv checkpoints/grape_simpler_sft checkpoints/grape_simpler_sft_dataset
CUDA_VISIBLE_DEVICES=3 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/steps_4000" \
  --lora_name "lora_004000"


CUDA_VISIBLE_DEVICES=3 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora_dpo.py \
  --vla_path "ZijianZhang/OpenVLA-7B-SFT-Simpler" \
  --run_path results/grape/adapter/openvla-7b+grape_simpler_dpos_dataset+b1+lr-2e-05+lora-r32+dropout-0.0/d1121_check
