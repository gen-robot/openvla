# task_name="maniskill"

# torchrun --standalone --nnodes 1 --nproc-per-node 3 vla-scripts/finetune_cobot.py \
#   --vla_path "openvla/openvla-7b" \
#   --data_root_dir datasets \
#   --dataset_name mani_skill_rlds_dataset \
#   --run_root_dir checkpoints/${task_name} \
#   --adapter_tmp_dir checkpoints/${task_name}/_tmp_adapter \
#   --lora_rank 64 \
#   --batch_size 4 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug True \
#   --wandb_project OpenVLA \
#   --wandb_entity fengg \
#   --save_steps 10000

task_name="cobot_rlds_dataset"

# OFT: PD + AC
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/debug/${task_name} \
  --use_proprio False \
  --use_film False \
  --num_images_in_input 1 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project Debug \
  --wandb_entity fengg \
  --max_steps 60_000 \
  --merge_lora_during_training True \
  --use_l1_regression False

exit 0

# OFT: PD + AC + L1
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --use_proprio False \
  --use_film False \
  --num_images_in_input 1 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA-SFT \
  --wandb_entity fengg \
  --max_steps 60_000 \
  --merge_lora_during_training True

# OFT: PD + AC + L1 + wrist
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --use_proprio False \
  --use_film False \
  --num_images_in_input 3 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA-SFT \
  --wandb_entity fengg \
  --max_steps 60_000 \
  --merge_lora_during_training True
