task_name="open_drawer_2"

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_cobot.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name cobot_future_dataset \
  --run_root_dir checkpoints/${task_name} \
  --adapter_tmp_dir checkpoints/${task_name}/_tmp_adapter \
  --lora_rank 64 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity fengg \
  --save_steps 10000