task_name="mani_skill_rlds_dataset"
use_film=False                      # if True, it will inject the language instruction into the visual encoder via FiLM
use_proprio=True                    # if True, it will use the proprioceptive sensor data, which would be useful for L1 regression or diffusion head
num_images_in_input=1               # the number of images in the input. if you want to use wrist images, set it to 3 or whatever you want.
####
# if use_l1_regression and use_diffusion are both False, 
# it will use the original discrete action head. 
# l1 and diffusion options cannot be both True at the same time.
####
use_l1_regression=False             # if True, it will use the L1 regression head
use_diffusion=False                 # if True, it will use the diffusion head
merge_lora_during_training=False    # if True, it will merge the LoRA weights during training, which will slightly increase the GPU memory usage
chunk_size=64                       # the number of actions to be predicted
use_parallel_decoding=True          # if you use a large chunk_size, make sure you have enabled parallel decoding in the model to increase the throughput
num_gpus=1

torchrun --standalone --nnodes 1 --nproc-per-node ${num_gpus} vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --use_proprio ${use_proprio} \
  --use_film ${use_film} \
  --num_images_in_input ${num_images_in_input} \
  --lora_rank 64 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA-SFT \
  --max_steps 200_000 \
  --merge_lora_during_training ${merge_lora_during_training} \
  --use_l1_regression ${use_l1_regression} \
  --use_diffusion ${use_diffusion} \
  --future_action_window_size ${chunk_size} \
  --use_parallel_decoding ${use_parallel_decoding}