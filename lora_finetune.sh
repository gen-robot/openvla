task_name="panda_simpler_sft_dataset"
CUDA_VISIBLE_DEVICES=5,6 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 50000 \
  --save_steps 2000 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project RLVLA \
  --wandb_entity pancake-wbw- \
  --version 1.1.0 \
  --lora_path "" \
  --save_optimizer True


# mv checkpoints/grape_simpler_sft checkpoints/grape_simpler_sft_dataset
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/1.0.0/steps_50000_bs_8" \
  --lora_name "lora_050000"

# "openvla/openvla-7b"
# "ZijianZhang/OpenVLA-7B-SFT-Simpler"


CUDA_VISIBLE_DEVICES=3 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora_dpo.py \
  --vla_path "ZijianZhang/OpenVLA-7B-SFT-Simpler" \
  --run_path results/grape/adapter/openvla-7b+grape_simpler_dpos_dataset+b1+lr-2e-05+lora-r32+dropout-0.0/d1121_check
