task_name="maniskill"

torchrun --standalone --nnodes 1 --nproc-per-node 3 vla-scripts/finetune-dpo.py \
  --vla_path "/nvme1n1/liangzhi/pretrained/openvla-7b+mani_skill+joint_pos/" \
  --chosen_traj_dir /nvme1n1/liangzhi/openvla_dataset/success_data_dir \
  --rejected_traj_dir /nvme1n1/liangzhi/openvla_dataset/fail_data_dir \
  --run_root_dir checkpoints/${task_name} \
  --adapter_tmp_dir checkpoints/${task_name}/_tmp_adapter \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity slzhta \
  --save_steps 10000
