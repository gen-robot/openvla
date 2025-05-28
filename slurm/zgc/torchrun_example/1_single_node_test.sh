#!/bin/bash

#SBATCH --job-name=pytorch_single_node         # 作业名称
#SBATCH --nodes=1                              # 单节点
#SBATCH --ntasks-per-node=1                    # 每个节点1个任务 (torchrun处理多GPU)
#SBATCH --gpus-per-node=8                      # 每个节点8张GPU
#SBATCH --cpus-per-task=32                     # 每个任务32个CPU核心
#SBATCH --mem=64G                              # 节点内存
#SBATCH --time=00:15:00                        # 运行时间限制
#SBATCH --output=logs/single_node_%j.out
#SBATCH --error=logs/single_node_%j.err

# --- 环境设置 ---
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=1

# --- 单节点配置 ---
export MASTER_ADDR=localhost
export MASTER_PORT=39501
export NNODES=1
export NPROC_PER_NODE=$SLURM_GPUS_PER_NODE

echo "=== Single-Node Distributed Training Setup ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "Current node: $(hostname)"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

echo "=== Environment Verification ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "NCCL available: $(python -c 'import torch; print(torch.distributed.is_nccl_available())')"

echo "=== Starting single-node torchrun ==="

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_debug.py

echo "Single-node job finished." 