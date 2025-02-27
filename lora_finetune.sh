task_name="grape_simpler_sft_dataset"

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_distributed.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "../datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 8 \
  --max_steps 800 \
  --save_steps 200 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project RLVLA \
  --wandb_entity hosnls

mv checkpoints/grape_simpler_sft checkpoints/grape_simpler_sft_dataset

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/merge_lora.py \
  --vla_path "openvla/openvla-7b" \
  --run_path "checkpoints/${task_name}/openvla-7b+${task_name}+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug" \
  --lora_name "lora_000800"
