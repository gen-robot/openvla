#!/bin/bash

#SBATCH --job-name=pytorch_torchrun_single     # 作业名称
#SBATCH --nodes=1                              # 请求1个节点用于测试
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=1                    # 每个节点只运行1个任务 (torchrun会处理多GPU)
#SBATCH --gpus-per-node=8                      # 每个节点请求8张GPU
#SBATCH --cpus-per-task=32                     # 每个任务分配32个CPU核心 (用于数据加载等)
#SBATCH --mem=64G                              # 每个节点内存 (可以根据需要调整)
#SBATCH --time=00:10:00                        # 作业运行时间限制 (小时:分钟:秒)
#SBATCH --output=logs/pytorch_torchrun_single_%j.out  # 标准输出文件，%j会被作业ID替换
#SBATCH --error=logs/pytorch_torchrun_single_%j.err   # 标准错误文件

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

# --- MASTER Address 和 Port 配置 (单节点简化) ---
export MASTER_ADDR=localhost
export MASTER_PORT=39504

# --- 计算分布式训练参数 ---
export NNODES=1                               # 单节点
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE    # 每个节点的进程数 (等于GPU数量)

echo "=== Single-Node Distributed Training Setup ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "Current node: $(hostname)"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

# 验证环境
echo "=== Environment Verification ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "NCCL available: $(python -c 'import torch.distributed; print(torch.distributed.is_nccl_available())')"

# --- 单节点torchrun (无需srun) ---
echo "=== Starting single-node torchrun ==="

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py

echo "Single-node job finished." 