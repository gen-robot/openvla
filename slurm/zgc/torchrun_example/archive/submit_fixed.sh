#!/bin/bash

#SBATCH --job-name=pytorch_torchrun_fixed      # 作业名称
#SBATCH --nodes=2                              # 请求2个节点
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=1                    # 每个节点只运行1个任务 (torchrun会处理多GPU)
#SBATCH --gpus-per-node=8                      # 每个节点请求8张GPU
#SBATCH --cpus-per-task=32                     # 每个任务分配32个CPU核心 (用于数据加载等)
#SBATCH --mem=64G                              # 每个节点内存 (可以根据需要调整)
#SBATCH --time=00:15:00                        # 作业运行时间限制 (小时:分钟:秒)
#SBATCH --output=logs/pytorch_torchrun_fixed_%j.out  # 标准输出文件，%j会被作业ID替换
#SBATCH --error=logs/pytorch_torchrun_fixed_%j.err   # 标准错误文件

# --- 环境设置 ---
# 激活你的conda环境 (如果使用conda)
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0

# 设置OMP线程数以避免CPU资源竞争
export OMP_NUM_THREADS=1

# --- MASTER Address 和 Port 配置 (修复版本) ---
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# 尝试多种方式获取master节点的IP地址
echo "=== Master Node Resolution ==="
echo "Master hostname from SLURM: $MASTER_NODE_HOSTNAME"

# 方法1: 使用getent hosts解析
MASTER_IP_GETENT=$(getent hosts $MASTER_NODE_HOSTNAME | awk '{print $1}' | head -n 1)
echo "Master IP via getent: $MASTER_IP_GETENT"

# 方法2: 使用nslookup解析 
MASTER_IP_NSLOOKUP=$(nslookup $MASTER_NODE_HOSTNAME | grep 'Address:' | tail -n 1 | awk '{print $2}')
echo "Master IP via nslookup: $MASTER_IP_NSLOOKUP"

# 方法3: 如果在主节点上，使用当前节点的bond0接口IP
if [ "$(hostname)" = "$MASTER_NODE_HOSTNAME" ]; then
    MASTER_IP_BOND0=$(ip addr show bond0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1)
    echo "Master IP via bond0 (current node): $MASTER_IP_BOND0"
fi

# 选择最佳的IP地址 (优先使用getent结果)
if [ -n "$MASTER_IP_GETENT" ]; then
    export MASTER_ADDR=$MASTER_IP_GETENT
    echo "Using getent result: $MASTER_ADDR"
elif [ -n "$MASTER_IP_BOND0" ]; then
    export MASTER_ADDR=$MASTER_IP_BOND0
    echo "Using bond0 result: $MASTER_ADDR"
elif [ -n "$MASTER_IP_NSLOOKUP" ]; then
    export MASTER_ADDR=$MASTER_IP_NSLOOKUP
    echo "Using nslookup result: $MASTER_ADDR"
else
    export MASTER_ADDR=$MASTER_NODE_HOSTNAME
    echo "Fallback to hostname: $MASTER_ADDR"
fi

export MASTER_PORT=39505  # 使用新端口避免冲突

# --- 计算分布式训练参数 ---
export NNODES=$SLURM_NNODES                    # 节点数量
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE     # 每个节点的进程数 (等于GPU数量)

echo "=== Distributed Training Setup ==="
echo "Master Node Hostname: $MASTER_NODE_HOSTNAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "Current node: $(hostname)"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# 网络连通性测试
echo "=== Network Connectivity Test ==="
echo "Testing ping to master node..."
ping -c 2 $MASTER_ADDR || echo "WARN: Cannot ping master node"

echo "Testing telnet to master port (if netcat available)..."
if command -v nc >/dev/null 2>&1; then
    timeout 5 nc -zv $MASTER_ADDR $MASTER_PORT || echo "Port $MASTER_PORT not yet open on $MASTER_ADDR"
fi

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

# 验证环境
echo "=== Environment Verification ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "NCCL available: $(python -c 'import torch; print(torch.distributed.is_nccl_available())')"

# --- 关键修复: 使用 srun 在所有节点上同时启动 torchrun ---
echo "=== Starting torchrun on all nodes simultaneously ==="

srun bash -c "
    # 设置每个节点的NODE_RANK
    export NODE_RANK=\$SLURM_NODEID
    
    echo \"[Node \$(hostname)] Starting torchrun with NODE_RANK=\$NODE_RANK\"
    echo \"[Node \$(hostname)] MASTER_ADDR=$MASTER_ADDR, MASTER_PORT=$MASTER_PORT\"
    echo \"[Node \$(hostname)] Will connect to $MASTER_ADDR:$MASTER_PORT\"
    
    torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=\$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py
"

echo "Job finished." 