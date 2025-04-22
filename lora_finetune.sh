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





# # from feng 
# # task_name="maniskill"
# # torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_cobot.py \
#   # --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
#   # --data_root_dir /nvme1n1/liangzhi/openvla_dataset/success_data_dir_small \
#   # --dataset_name mani_skill_rlds_dataset \
#   # --run_root_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-sft \
#   # --adapter_tmp_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-sft/_tmp_adapter \
#   # --lora_rank 64 \
#   # --batch_size 4 \
#   # --grad_accumulation_steps 1 \
#   # --learning_rate 5e-4 \
#   # --image_aug True \
#   # --wandb_project OpenVLA \
#   # --wandb_entity slzhta \
#   # --save_steps 10000


# task_name="cobot_rlds_dataset"

# # OFT: PD + AC
# torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
#   --vla_path "openvla/openvla-7b" \
#   --data_root_dir datasets \
#   --dataset_name ${task_name} \
#   --run_root_dir checkpoints/debug/${task_name} \
#   --use_proprio False \
#   --use_film False \
#   --num_images_in_input 3 \
#   --lora_rank 32 \
#   --batch_size 1 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug True \
#   --wandb_project Debug \
#   --wandb_entity fengg \
#   --max_steps 60_000 \
#   --merge_lora_during_training True \
#   --use_l1_regression False \
#   --window_size 3

# exit 0

# # OFT: PD + AC + L1
# torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
#   --vla_path "openvla/openvla-7b" \
#   --data_root_dir datasets \
#   --dataset_name ${task_name} \
#   --run_root_dir checkpoints/${task_name} \
#   --use_proprio False \
#   --use_film False \
#   --num_images_in_input 1 \
#   --lora_rank 32 \
#   --batch_size 1 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug True \
#   --wandb_project OpenVLA-SFT \
#   --wandb_entity fengg \
#   --max_steps 60_000 \
#   --merge_lora_during_training True

# # OFT: PD + AC + L1 + wrist
# torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
#   --vla_path "openvla/openvla-7b" \
#   --data_root_dir datasets \
#   --dataset_name ${task_name} \
#   --run_root_dir checkpoints/${task_name} \
#   --use_proprio False \
#   --use_film False \
#   --num_images_in_input 3 \
#   --lora_rank 32 \
#   --batch_size 1 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug True \
#   --wandb_project OpenVLA-SFT \
#   --wandb_entity fengg \
#   --max_steps 60_000 \
#   --merge_lora_during_training True
