task_name="maniskill-single"
bs=$1

python vla-scripts/finetune_cobot-single.py \
  --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
  --data_root_dir /nvme1n1/liangzhi/openvla_dataset/iterative-1 \
  --dataset_name mani_skill_rlds_dataset \
  --run_root_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-sft-bs${bs}-full-1 \
  --adapter_tmp_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-sft-bs${bs}-full-1/_tmp_adapter \
  --lora_rank 64 \
  --batch_size ${bs} \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity slzhta \
  --save_steps 10000