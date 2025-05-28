#!/bin/bash

#SBATCH --job-name=vla_finetune_test           # 作业名称
#SBATCH --nodes=2                              # 请求2个节点用于测试
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=1                    # 每个节点只运行1个任务 (torchrun会处理多GPU)
#SBATCH --gpus-per-node=8                      # 每个节点请求8张GPU
#SBATCH --cpus-per-task=32                     # 每个任务分配32个CPU核心 (用于数据加载等)
#SBATCH --mem=64G                              # 每个节点内存 (可以根据需要调整)
#SBATCH --time=00:30:00                        # 作业运行时间限制 (30分钟足够测试)
#SBATCH --output=logs/vla_finetune_test_%j.out # 标准输出文件，%j会被作业ID替换
#SBATCH --error=logs/vla_finetune_test_%j.err  # 标准错误文件

# --- 环境设置 ---
# 激活你的conda环境 (如果使用conda)
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数 (模拟finetune.py的环境)
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0

# 确保你的环境中安装了 PyTorch, accelerate 等 (finetune.py需要的依赖)

# --- MASTER Address 和 Port 配置 ---
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE_HOSTNAME
export MASTER_PORT=39502  # 使用不同端口避免冲突

echo "=== VLA Finetune Test Setup ==="
echo "Master Node Hostname: $MASTER_NODE_HOSTNAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# --- 计算分布式训练参数 ---
export NNODES=$SLURM_NNODES                    # 节点数量
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE     # 每个节点的进程数 (等于GPU数量)
export NODE_RANK=$SLURM_NODEID                 # 当前节点的排名

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "NODE_RANK: $NODE_RANK"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

# 验证依赖项
echo "=== Verifying Dependencies ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
python -c "import torch.distributed; print(f'Distributed available: {torch.distributed.is_available()}')"
python -c "import torch.distributed; print(f'NCCL available: {torch.distributed.is_nccl_available()}')"

echo "=== Starting VLA-like Distributed Training Test ==="
echo "This test validates the torchrun setup for finetune.py"
echo "Testing with:"
echo "  - ${NNODES} nodes, ${NPROC_PER_NODE} GPUs per node"
echo "  - Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "  - FSDP model sharding"
echo "  - Gradient accumulation"
echo "  - Mock VLA model architecture"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    test_finetune_setup.py

echo "=== Test Complete ==="
echo "If this test passes successfully, your torchrun setup should work with finetune.py" 