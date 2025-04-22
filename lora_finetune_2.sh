task_name="cobot_rlds_dataset"

# # OFT: PD + AC + L1 + wrist + proprio
# torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
#   --vla_path "openvla/openvla-7b" \
#   --data_root_dir datasets \
#   --dataset_name ${task_name} \
#   --run_root_dir checkpoints/${task_name} \
#   --use_proprio True \
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

# OFT: PD + AC + L1 + wrist + proprio + film
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --use_proprio True \
  --use_film True \
  --num_images_in_input 3 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA-SFT \
  --wandb_entity fengg \
  --max_steps 200_000 \
  --merge_lora_during_training False \
  --use_l1_regression False \
  --use_diffusion True
  # --window_size 3

# OFT: PD + AC + diffusion + wrist + proprio + film
# torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
#   --vla_path "openvla/openvla-7b" \
#   --data_root_dir datasets \
#   --dataset_name ${task_name} \
#   --run_root_dir checkpoints/${task_name} \
#   --use_proprio True \
#   --use_film True \
#   --num_images_in_input 3 \
#   --lora_rank 32 \
#   --batch_size 1 \
#   --grad_accumulation_steps 1 \
#   --learning_rate 5e-4 \
#   --image_aug True \
#   --wandb_project OpenVLA-SFT \
#   --wandb_entity fengg \
#   --max_steps 60_000 \
#   --merge_lora_during_training True \
#   --use_l1_regression False \
#   --use_diffusion True