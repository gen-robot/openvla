task_name="cobot_rlds_dataset"

torchrun --standalone --nnodes 1 --nproc-per-node 6 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --use_proprio True \
  --num_images_in_input 3 \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity fengg 