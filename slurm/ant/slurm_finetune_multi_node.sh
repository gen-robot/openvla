#!/bin/bash

#SBATCH --job-name=gf_vla                          # 作业名称
#SBATCH --nodes=2                                  # 3个节点 (3×8=24 GPUs total)
#SBATCH --ntasks-per-node=1                        # 每个节点1个任务 (torchrun处理多GPU)
#SBATCH --gpus-per-node=8                          # 每个节点8张GPU
#SBATCH --cpus-per-task=32                         # 每个任务32个CPU核心
#SBATCH --mem=256G                                 # 节点内存 (VLA需要更多内存)
#SBATCH --time=24:00:00                            # 运行时间限制 (24小时)
#SBATCH --output=logs/vla_finetune_multi_%j.out
#SBATCH --error=logs/vla_finetune_multi_%j.err

# 设置NCCL参数 (与测试中验证的相同)
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=1

export SIF_IMAGE="/storage/openpsi/images/mlaas-eai-v2.6.sif"

# --- VLA Fine-tuning 配置 (基于原始 finetune.sh) ---
task_name="libero_object_no_noops"
use_film=False
use_proprio=False
num_images_in_input=1
use_l1_regression=False
use_diffusion=False
merge_lora_during_training=False
num_actions_chunk=1
use_parallel_decoding=False
is_debug=False
enable_cot=True
use_lora=True
cot_full=True
resume=False
vla_path=openvla/openvla-7b
cot_tags="move_reason,move"

# 多节点配置
project_name="VLA-Reasoning"

# --- 多节点配置 ---
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE_HOSTNAME

export MASTER_PORT=39522
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "=== VLA Fine-tuning Setup (Multi-Node) ==="
echo "Task: $task_name"
echo "VLA Path: $vla_path"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Master hostname: $MASTER_NODE_HOSTNAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Total GPUs: $(($NNODES * $NPROC_PER_NODE))"

# 网络连通性测试
echo "=== Network Connectivity Test ==="
ping -c 1 $MASTER_ADDR && echo "✓ Master node ping successful" || echo "✗ Master node ping failed"

cd /storage/openpsi/users/gaofeng/openvla/

echo "=== Environment Verification ==="
# echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
# echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
# echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
# echo "NCCL available: $(python -c 'import torch; print(torch.distributed.is_nccl_available())')"

# 创建日志目录
mkdir -p logs

echo "=== Starting Multi-Node VLA Fine-tuning ==="

# 使用srun在所有节点同时启动torchrun (基于验证的方法)
srun --mpi=pmi2 \
    singularity exec --nv \
    --pid \
    --writable-tmpfs \
    --no-home \
    --bind /storage:/storage \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-862bc023bdff4309bad1e6cc319369cdb354b4ac \
    $SIF_IMAGE \
    bash -c "
    export NODE_RANK=\$SLURM_NODEID
    
    # 强制设置正确的环境变量
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT

    echo $http_proxy $https_proxy $all_proxy
    export http_proxy=; export https_proxy=; export HTTP_PROXY=; export HTTPS_PROXY=; export all_proxy=; export ALL_PROXY=
    echo $http_proxy $https_proxy $all_proxy
    
    echo \"[Node \$(hostname) - Node Rank \$NODE_RANK] Starting VLA fine-tuning\"
    echo \"[Node \$(hostname)] MASTER_ADDR=\$MASTER_ADDR, MASTER_PORT=\$MASTER_PORT\"

    # Setup directories
    mkdir -p /code/EmbodiedAgent/embodied_agent/third_party/vla/
    cd /code/EmbodiedAgent/embodied_agent/third_party/vla/
    ln -sf /storage/openpsi/users/gaofeng/openvla

    cd /storage/openpsi/users/gaofeng/openvla/
    source '/opt/conda/etc/profile.d/conda.sh'
    conda activate embodied
    conda info --env
    
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=\$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        vla-scripts/finetune.py \
        --vla_path ${vla_path} \
        --data_root_dir datasets/libero_data \
        --dataset_name ${task_name} \
        --run_root_dir checkpoints/${task_name} \
        --use_proprio ${use_proprio} \
        --use_film ${use_film} \
        --num_images_in_input ${num_images_in_input} \
        --use_lora ${use_lora} \
        --lora_rank 32 \
        --batch_size 2 \
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
        --use_val_set False \
        --save_freq 5000 \
        --val_freq 1000
"

echo "Multi-node VLA fine-tuning completed." 