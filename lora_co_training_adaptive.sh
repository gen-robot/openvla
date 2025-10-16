set_name="panda_co_training_0.5"
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
num_actions_chunk=$4                 # the number of actions to be predicted
use_parallel_decoding=True         # if you use a large chunk_size, make sure you have enabled parallel decoding in the model to increase the throughput
is_debug=False
enable_cot=False
use_lora=True
cot_full=False
vla_path=$1
cot_tags="move_reason,move"
task_name=$2
shuffle_buffer_size=$3
env_id=$5
adaptive_ratio=$6
fixed_ratio_value=$7

# if is_debug is True, set num_gpus to 1, set project name to OpenVLA-debug
if [ ${is_debug} = True ]; then
    num_gpus=1
    project_name="OpenVLA-debug"
elif [ ${enable_cot} = True ]; then
    num_gpus=2
    project_name="VLA-Reasoning"
else
    num_gpus=3
    project_name="VLA-Co-Training"
fi

if [ ${env_id} = "PandaPutOnPlateInScene25Simple-v1" ]; then
    image_cut_start=40
elif [ ${env_id} = "PandaOpenDrawer-v1" ]; then
    image_cut_start=160
else
    image_cut_start=0
fi
echo $vla_path
torchrun --standalone --nnodes 1 --nproc-per-node ${num_gpus} vla-scripts/finetune_amcr.py \
    --vla_path ${vla_path} \
    --data_root_dir /mnt/public/shiliangzhi/openvla-datasets/franka_panda_${task_name}-sim_real_co_training \
    --dataset_statistics_path /mnt/public/shiliangzhi/openvla-datasets/dataset_statistics/${task_name} \
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
    --num_actions_chunk ${num_actions_chunk} \
    --save_freq 1000 \
    --shuffle_buffer_size ${shuffle_buffer_size} \
    --env_id ${env_id} \
    --obs_mode rgb+segmentation \
    --image_cut_start ${image_cut_start} \
    --adaptive_ratio ${adaptive_ratio} \
    --fixed_ratio_value ${fixed_ratio_value}