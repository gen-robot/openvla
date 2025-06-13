#!/bin/bash

#SBATCH --job-name=pytorch_torchrun_debug      # 作业名称
#SBATCH --nodes=1                              # 请求1个节点用于调试
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=1                    # 每个节点只运行1个任务 (torchrun会处理多GPU)
#SBATCH --gpus-per-node=2                      # 只使用2张GPU进行调试
#SBATCH --cpus-per-task=8                      # 每个任务分配8个CPU核心
#SBATCH --mem=32G                              # 每个节点内存
#SBATCH --time=00:10:00                        # 作业运行时间限制
#SBATCH --output=logs/pytorch_torchrun_debug_%j.out  # 标准输出文件
#SBATCH --error=logs/pytorch_torchrun_debug_%j.err   # 标准错误文件

echo "=== Debug Session Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

# --- 环境设置 ---
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数 (简化版用于调试)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0

# 设置OMP线程数
export OMP_NUM_THREADS=1

# --- 简化的分布式参数 ---
export MASTER_ADDR=localhost
export MASTER_PORT=39505
export NNODES=1
export NPROC_PER_NODE=2  # 只使用2个GPU进行调试

echo "=== Environment Check ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

# 验证环境
echo "=== Dependency Verification ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "PyTorch import failed!"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "CUDA check failed!"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" || echo "GPU count check failed!"
python -c "import torch.distributed; print(f'Distributed: {torch.distributed.is_available()}')" || echo "Distributed check failed!"
python -c "import torch.distributed; print(f'NCCL: {torch.distributed.is_nccl_available()}')" || echo "NCCL check failed!"

echo "=== Starting Debug Torchrun ==="
echo "Command: torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_debug.py"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_debug.py

echo "=== Debug Session Finished ==="
echo "Exit code: $?" 