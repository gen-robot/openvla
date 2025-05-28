#!/bin/bash

#SBATCH --job-name=pytorch_torchrun_simple     # 作业名称
#SBATCH --nodes=2                              # 请求2个节点
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=1                    # 每个节点只运行1个任务 (torchrun会处理多GPU)
#SBATCH --gpus-per-node=2                      # 每个节点请求2张GPU (减少复杂度)
#SBATCH --cpus-per-task=8                      # 每个任务分配8个CPU核心
#SBATCH --mem=32G                              # 每个节点内存
#SBATCH --time=00:10:00                        # 作业运行时间限制
#SBATCH --output=logs/pytorch_torchrun_simple_%j.out
#SBATCH --error=logs/pytorch_torchrun_simple_%j.err

# --- 环境设置 ---
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=1

# --- 简化的MASTER Address配置 ---
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

echo "=== Node Information ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Master hostname: $MASTER_NODE_HOSTNAME"
echo "Current hostname: $(hostname)"
echo "Current node IP on bond0: $(ip addr show bond0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)"

# 修复: 直接使用srun在master节点上获取其bond0 IP地址
echo "=== Resolving Master Node IP ==="
export MASTER_ADDR=$(srun --nodes=1 --nodelist=$MASTER_NODE_HOSTNAME bash -c "ip addr show bond0 | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1")

echo "Master node bond0 IP: $MASTER_ADDR"

# 验证IP地址格式
if [[ $MASTER_ADDR =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "✓ Valid IP address format: $MASTER_ADDR"
else
    echo "✗ Invalid IP format, falling back to hostname"
    export MASTER_ADDR=$MASTER_NODE_HOSTNAME
fi

export MASTER_PORT=39507
export NNODES=$SLURM_NNODES
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "=== Final Configuration ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# 网络测试
echo "=== Network Test ==="
ping -c 1 $MASTER_ADDR && echo "✓ Ping successful" || echo "✗ Ping failed"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

echo "=== Environment Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "=== Starting Multi-Node Training ==="

# 使用srun同时在所有节点启动torchrun
srun bash -c "
    export NODE_RANK=\$SLURM_NODEID
    
    echo \"[Node \$(hostname) - Rank \$NODE_RANK] Starting torchrun\"
    echo \"[Node \$(hostname) - Rank \$NODE_RANK] Connecting to $MASTER_ADDR:$MASTER_PORT\"
    
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

echo "Multi-node job completed." 