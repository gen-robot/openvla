set_name=$3
use_film=False                      # if True, it will inject the language instruction into the visual encoder via FiLM
use_proprio=False                    # if True, it will use the proprioceptive sensor data, which would be useful for L1 regression or diffusion head
num_images_in_input=1               # the number of images in the input. if you want to use wrist images, set it to 3 or whatever you want.
####
# if use_l1_regression and use_diffusion are both False, 
# it will use the original discrete action head. 
# l1 and diffusion options cannot be both True at the same time.
####
use_l1_regression=True              # if True, it will use the L1 regression head
use_diffusion=False                 # if True, it will use the diffusion head
merge_lora_during_training=False    # if True, it will merge the LoRA weights during training, which will slightly increase the GPU memory usage
use_parallel_decoding=True         # if you use a large chunk_size, make sure you have enabled parallel decoding in the model to increase the throughput
is_debug=False
enable_cot=False
use_lora=True
cot_full=False
vla_path=$1
cot_tags="move_reason,move"
task_name=$2
shuffle_buffer_size=100000

num_gpus=1
project_name="VLA-SFT"

python vla-scripts/dataset_statistics_pre_compute.py \
    --vla_path ${vla_path} \
    --data_root_dir /mnt/public/shiliangzhi/openvla-datasets/franka_panda_${task_name}-dataset_gen \
    --dataset_name ${set_name} \
    --run_root_dir checkpoints/${task_name}-${set_name} \
    --use_proprio ${use_proprio} \
    --use_film ${use_film} \
    --num_images_in_input ${num_images_in_input} \
    --use_lora ${use_lora} \
    --lora_rank 32 \
    --batch_size 16 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug False \
    --wandb_project ${project_name} \
    --max_steps 100_000 \
    --merge_lora_during_training ${merge_lora_during_training} \
    --use_l1_regression ${use_l1_regression} \
    --use_diffusion ${use_diffusion} \
    --use_parallel_decoding ${use_parallel_decoding} \
    --enable_cot ${enable_cot} \
    --save_freq 1000 \
    --shuffle_buffer_size ${shuffle_buffer_size}