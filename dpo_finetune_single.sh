task_name="maniskill"
gas=$1

python vla-scripts/finetune-dpo-single.py \
  --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
  --chosen_traj_dir /nvme1n1/liangzhi/openvla_dataset/success_data_dir \
  --rejected_traj_dir /nvme1n1/liangzhi/openvla_dataset/fail_data_dir \
  --run_root_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-${gas} \
  --adapter_tmp_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-${gas}/_tmp_adapter \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps ${gas} \
  --learning_rate 2e-5 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity slzhta \
  --save_steps 10000
