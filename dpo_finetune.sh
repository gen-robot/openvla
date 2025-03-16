task_name="maniskill_multi"
gas=$1
use_lora=$2

torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune-dpo.py \
  --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
  --chosen_traj_dir /nvme1n1/liangzhi/openvla_dataset/success_data_dir_small \
  --rejected_traj_dir /nvme1n1/liangzhi/openvla_dataset/fail_data_dir_small \
  --run_root_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-gas${gas}-sc${sc} \
  --adapter_tmp_dir /nvme1n1/liangzhi/openvla_checkpoint/${task_name}-gas${gas}-sc${sc}/_tmp_adapter \
  --use_lora ${use_lora} \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps ${gas} \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project OpenVLA \
  --wandb_entity slzhta \
  --save_steps 10000
