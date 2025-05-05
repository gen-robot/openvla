# 80GB: 20
# 40GB: 8
task_name="spc148f"
CUDA_VISIBLE_DEVICES=3,4 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/nvme_data/bingwen/Documents/arm_ws/SimplerEnv/datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --unnorm_key="bridge_orig" \
  --lora_rank 32 \
  --batch_size 20 \
  --max_steps 2100 \
  --eval_steps 50 \
  --save_steps "0,50,100,200,300,500,750,1000,2000,2100" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "RLVLA_sft" \
  --wandb_entity hosnls

vla_path="openvla/openvla-7b"
task_name="bingwen"
CUDA_VISIBLE_DEVICES=2,6 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "${vla_path}" \
  --data_root_dir "../datasets" \
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
task_name="spc148f"
CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/steps_2000" \
  --lora_name "lora_002000"


# new jijia

#vla_path="openvla/openvla-7b"
vla_path="checkpoints/spc148f/steps_2000/merged_002000"
task_name="s5120c5"
CUDA_VISIBLE_DEVICES=2,6 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "${vla_path}" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 20 \
  --max_steps 4000 \
  --eval_steps 100 \
  --save_steps "0,50,100,200,300,500,750,1000,1500,2000,3000,5000,7500,10000,12500,15000,17500,20000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "RLVLA_sft" \
  --wandb_entity hosnls

vla_path="openvla/openvla-7b"
CUDA_VISIBLE_DEVICES=2,6 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "${vla_path}" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 20 \
  --max_steps 4000 \
  --eval_steps 100 \
  --save_steps "0,50,100,200,300,500,750,1000,1500,2000,3000,5000,7500,10000,12500,15000,17500,20000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "RLVLA_sft" \
  --wandb_entity hosnls


#mv checkpoints/grape_simpler_sft checkpoints/grape_simpler_sft_dataset
task_name="spc148f"
CUDA_VISIBLE_DEVICES=7 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/steps_2000" \
  --lora_name "lora_002000"
CUDA_VISIBLE_DEVICES=7 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/steps_2100" \
  --lora_name "lora_002000"


CUDA_VISIBLE_DEVICES=0 \
torchrun --standalone --nnodes=1 --nproc-per-node 1 vla-scripts/finetune_grape.py \
  --vla_path "openvla/openvla-7b" \
  --dataset_s_name "grape_simpler_dpos_dataset" \
  --dataset_f_name "grape_simpler_dpof_dataset" \
  --traj_dir "../datasets" \
  --run_root_dir "results/grape/root" \
  --adapter_tmp_dir "results/grape/adapter" \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project "rlvla_sft" \
  --wandb_entity "hosnls" \
  --save_steps 1000

# for 40GB training
train_cuda_device=1
ref_cuda_device=2
CUDA_VISIBLE_DEVICES=$train_cuda_device \
torchrun --standalone --nnodes=1 --nproc-per-node=1 vla-scripts/finetune_grape_pair.py \
  --vla_path "openvla/openvla-7b" \
  --dataset_s_name "grape_simpler_dpos_dataset" \
  --dataset_f_name "grape_simpler_dpof_dataset" \
  --traj_dir "../datasets" \
  --run_root_dir "results/grape/root" \
  --adapter_tmp_dir "results/grape/adapter" \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project "rlvla_sft" \
  --wandb_entity "hosnls" \
  --save_steps 1000 \
  --ref_cuda_device $ref_cuda_device

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
