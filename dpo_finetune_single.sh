task_name="maniskill"
gas=$1
ul=$2

python vla-scripts/finetune-dpo-single.py \
  --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
  --chosen_traj_dir /nvme1n1/liangzhi/openvla_dataset/success_data_dir_small \
  --rejected_traj_dir /nvme1n1/liangzhi/openvla_dataset/fail_data_dir_small \
  --run_root_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-gas${gas}-lora${ul} \
  --adapter_tmp_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-gas${gas}-lora${ul}/_tmp_adapter \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps ${gas} \
  --use_lora ${ul} \
  --learning_rate 2e-5 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity slzhta \
  --save_steps 10000
