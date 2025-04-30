task_name="libero_object_no_noops"
use_film=False                      # if True, it will inject the language instruction into the visual encoder via FiLM
use_proprio=False                    # if True, it will use the proprioceptive sensor data, which would be useful for L1 regression or diffusion head
num_images_in_input=1               # the number of images in the input. if you want to use wrist images, set it to 3 or whatever you want.
####
# if use_l1_regression and use_diffusion are both False, 
# it will use the original discrete action head. 
# l1 and diffusion options cannot be both True at the same time.
####
use_l1_regression=False              # if True, it will use the L1 regression head
use_diffusion=False                 # if True, it will use the diffusion head
merge_lora_during_training=False    # if True, it will merge the LoRA weights during training, which will slightly increase the GPU memory usage
num_actions_chunk=1                 # the number of actions to be predicted
use_parallel_decoding=False         # if you use a large chunk_size, make sure you have enabled parallel decoding in the model to increase the throughput
is_debug=True
enable_cot=True
use_lora=False
cot_tags="move_reason,move"

# if is_debug is True, set num_gpus to 1, set project name to OpenVLA-debug
if [ ${is_debug} = True ]; then
    num_gpus=1
    project_name="OpenVLA-debug"
# elif [ ${enable_cot} = True ]; then
#     num_gpus=8
#     project_name="VLA-Reasoning"
else
    num_gpus=4
    project_name="VLA-Reasoning"
fi
# openvla-ecot/ecot-openvla-7b-oxe \
torchrun --standalone --nnodes 1 --nproc-per-node ${num_gpus} vla-scripts/finetune.py \
  --vla_path /nvme_data/liangzhi/pretrained/ecot-openvla-7b-oxe/ \
  --data_root_dir /nvme_data/liangzhi/openvla_dataset/libero_dataset/libero_object_original \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --use_proprio ${use_proprio} \
  --use_film ${use_film} \
  --num_images_in_input ${num_images_in_input} \
  --use_lora ${use_lora} \
  --lora_rank 1 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project ${project_name} \
  --max_steps 100_000 \
  --merge_lora_during_training ${merge_lora_during_training} \
  --use_l1_regression ${use_l1_regression} \
  --use_diffusion ${use_diffusion} \
  --use_parallel_decoding ${use_parallel_decoding} \
  --enable_cot ${enable_cot} \
  --num_actions_chunk ${num_actions_chunk} \
  --use_val_set True \
  --save_freq 5000 \
  --val_freq 1000 
#   --cot_tags ${cot_tags}