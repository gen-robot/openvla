GPU_LIST=(4 5 6 7)
NUM_GPUS=${#GPU_LIST[@]}
NUM_MAX_PROCESSES=16
dataset_name=$1
data_dir=$2
mode=$3

# mode should be one of the following:
# "reasoning", "gripper", "bboxes", "descriptions"
if [ "$mode" != "reasoning" ] && [ "$mode" != "gripper" ] && [ "$mode" != "bboxes" ] && [ "$mode" != "descriptions" ] && [ "$mode" != "object_lists" ]; then
    echo "Invalid mode: $mode"
    exit 1
fi
# set NUM_PROCESSES to NUM_MAX_PROCESSES for gripper and reasoning, NUM_GPUS for bboxes and descriptions
if [ "$mode" == "gripper" ] || [ "$mode" == "reasoning" ] || [ "$mode" == "object_lists" ]; then
    NUM_PROCESSES=$NUM_MAX_PROCESSES
else
    NUM_PROCESSES=$NUM_GPUS
fi

session_name="gen_${mode}_${dataset_name}"

# check if tmux session exists, if not create it
if ! tmux has-session -t $session_name 2>/dev/null; then
    # create a tmux session with NUM_GPUS windows
    tmux new-session -s $session_name -n $session_name -d

    # create NUM_GPUS windows inside the session
    for i in $(seq 0 $((NUM_PROCESSES - 1))); do
        # window 0 is already created, rename it
        if [ $i -eq 0 ]; then
            tmux rename-window -t $session_name:$i "${session_name}_${i}"
        else
            tmux new-window -t $session_name:$i -n "${session_name}_${i}"
        fi
        tmux send-keys -t $session_name:$i "conda activate embodied; start_proxy" C-m
    done
else
    read -p "Session already exists. Continue with execution? (y/n) " choice
    if [[ $choice != "y" && $choice != "Y" ]]; then
        echo "Exiting..."
        exit 0
    fi
fi

# run the commands in the windows
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    if [ "$mode" == "reasoning" ]; then
        tmux send-keys -t $session_name:$i "python full_reasonings.py --id $i --splits $NUM_PROCESSES --dataset_name $dataset_name --data_dir $data_dir" C-m
    elif [ "$mode" == "gripper" ]; then
        tmux send-keys -t $session_name:$i "python gripper_positions_gemini.py --id $i --splits $NUM_PROCESSES --dataset_name $dataset_name --data_dir $data_dir" C-m
    elif [ "$mode" == "bboxes" ]; then
        tmux send-keys -t $session_name:$i "CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} python generate_bboxes.py --id $i --splits $NUM_PROCESSES --dataset_name $dataset_name --data_dir $data_dir --visualize" C-m
    elif [ "$mode" == "descriptions" ]; then
        tmux send-keys -t $session_name:$i "python generate_descriptions.py --id $i --splits $NUM_PROCESSES --dataset_name $dataset_name --data_dir $data_dir --gpu ${GPU_LIST[$i]}" C-m
    elif [ "$mode" == "object_lists" ]; then
        tmux send-keys -t $session_name:$i "python generate_object_lists.py --id $i --splits $NUM_PROCESSES --dataset_name $dataset_name --data_dir $data_dir" C-m
    fi
done
