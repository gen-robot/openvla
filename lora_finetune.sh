# 80GB: 20
# 40GB: 8

task_name="spc148f"

CUDA_VISIBLE_DEVICES=3,4 \
torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune_distributed.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "../datasets" \
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

CUDA_VISIBLE_DEVICES=3 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora_dpo.py \
  --vla_path "ZijianZhang/OpenVLA-7B-SFT-Simpler" \
  --run_path results/grape/adapter/openvla-7b+grape_simpler_dpos_dataset+b1+lr-2e-05+lora-r32+dropout-0.0/d1121_check
