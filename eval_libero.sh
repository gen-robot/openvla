mode=$1
ckpt_path=$2
initial_state_id=$3
num_trials_per_task=$4
save_unit_id=$5
start_reasoning=$6

save_dir="/nvme_data/liangzhi/datasets/libero-eval-save"

# mode should be one of: spatial, object, goal, 10
if [ "$mode" != "spatial" ] && [ "$mode" != "object" ] && [ "$mode" != "goal" ] && [ "$mode" != "10" ]; then
  echo "Invalid mode: $mode"
  exit 1
fi

# if oft in ckpt_path, then useoft
# if [[ "$ckpt_path" == *"oft"* ]]; then
#   python experiments/robot/libero/run_libero_eval.py \
#     --task_suite_name libero_${mode} \
#     --pretrained_checkpoint $ckpt_path \
#     --use_parallel_decoding True \
#     --use_l1_regression True \
#     --use_diffusion False \
#     --use_film False \
#     --num_images_in_input 2 \
#     --use_proprio True \
#     --center_crop True \
#     --num_open_loop_steps 8
if [[ "$ckpt_path" == *"cot"* ]]; then
  python experiments/robot/libero/run_libero_eval.py \
    --task_suite_name libero_${mode} \
    --pretrained_checkpoint $ckpt_path \
    --use_parallel_decoding False \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --center_crop True \
    --num_open_loop_steps 1 \
    --num_actions_chunk 1 \
    --enable_cot True \
    --use_gemini_cot False \
    --initial_state_id ${initial_state_id} \
    --num_trials_per_task ${num_trials_per_task} \
    --save_data_dir ${save_dir} \
    --save_unit_id ${save_unit_id} \
    --start_reasoning ${start_reasoning}
else
  python experiments/robot/libero/run_libero_eval.py \
    --task_suite_name libero_${mode} \
    --pretrained_checkpoint $ckpt_path \
    --use_parallel_decoding False \
    --use_l1_regression False \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 1 \
    --use_proprio False \
    --center_crop True \
    --num_open_loop_steps 1 \
    --num_actions_chunk 1
fi