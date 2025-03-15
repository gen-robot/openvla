task_name="maniskill"

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_cobot.py \
  --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
  --data_root_dir /nvme1n1/liangzhi/openvla_dataset/success_data_dir_small \
  --dataset_name mani_skill_rlds_dataset \
  --run_root_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-sft \
  --adapter_tmp_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-sft/_tmp_adapter \
  --lora_rank 64 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity slzhta \
  --save_steps 10000