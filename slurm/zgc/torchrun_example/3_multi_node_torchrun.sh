#!/bin/bash

#SBATCH --job-name=pytorch_multi_node_torchrun # 作业名称
#SBATCH --nodes=2                              # 2个节点
#SBATCH --exclude=pm-9eea0003                  # 排除故障节点
#SBATCH --ntasks-per-node=1                    # 每个节点1个任务 (torchrun处理多GPU)
#SBATCH --gpus-per-node=2                      # 每个节点2张GPU
#SBATCH --cpus-per-task=8                      # 每个任务8个CPU核心
#SBATCH --mem=32G                              # 节点内存
#SBATCH --time=00:10:00                        # 运行时间限制
#SBATCH --output=logs/multi_node_torchrun_%j.out
#SBATCH --error=logs/multi_node_torchrun_%j.err

# --- 环境设置 ---
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=1

# --- 多节点配置 ---
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# 手动IP映射
case $MASTER_NODE_HOSTNAME in
    "pm-9eea")
        export MASTER_ADDR="10.3.27.3"
        ;;
    "pm-9eea0001")
        export MASTER_ADDR="10.3.27.4"
        ;;
    "pm-9eea0002")
        export MASTER_ADDR="10.3.27.5"
        ;;
    *)
        export MASTER_ADDR=$MASTER_NODE_HOSTNAME
        ;;
esac

export MASTER_PORT=39513
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "=== Multi-Node Distributed Training Setup (torchrun) ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Master hostname: $MASTER_NODE_HOSTNAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# 网络连通性测试
echo "=== Network Connectivity Test ==="
ping -c 1 $MASTER_ADDR && echo "✓ Master node ping successful" || echo "✗ Master node ping failed"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

echo "=== Environment Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "=== Starting Multi-Node Training (torchrun) ==="

# 使用srun在所有节点同时启动torchrun
srun bash -c "
    export NODE_RANK=\$SLURM_NODEID
    
    echo \"[Node \$(hostname) - Node Rank \$NODE_RANK] Starting torchrun\"
    echo \"[Node \$(hostname)] Connecting to $MASTER_ADDR:$MASTER_PORT\"
    
    # 强制设置环境变量，防止torchrun覆盖
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=\$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train_debug.py
"

echo "Multi-node torchrun job completed." 