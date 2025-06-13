#!/bin/bash

#SBATCH --job-name=pytorch_multi_node_srun     # 作业名称
#SBATCH --nodes=2                              # 2个节点
#SBATCH --exclude=pm-9eea0003                  # 排除故障节点
#SBATCH --ntasks-per-node=2                    # 每个节点2个任务 (对应2个GPU)
#SBATCH --gpus-per-node=2                      # 每个节点2张GPU
#SBATCH --cpus-per-task=4                      # 每个任务4个CPU核心
#SBATCH --mem=32G                              # 节点内存
#SBATCH --time=00:10:00                        # 运行时间限制
#SBATCH --output=logs/multi_node_srun_%j.out
#SBATCH --error=logs/multi_node_srun_%j.err

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

# 手动IP映射 (基于观察到的模式)
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

export MASTER_PORT=39512
export WORLD_SIZE=4  # 2 nodes × 2 GPUs = 4 total processes

echo "=== Multi-Node Distributed Training Setup (srun) ==="
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Master hostname: $MASTER_NODE_HOSTNAME"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

# 网络连通性测试
echo "=== Network Connectivity Test ==="
ping -c 1 $MASTER_ADDR && echo "✓ Master node ping successful" || echo "✗ Master node ping failed"

# 测试从所有节点的连通性
echo "=== Cross-Node Connectivity Test ==="
srun bash -c "echo \"[Node \$(hostname)] Testing ping to master $MASTER_ADDR\"; ping -c 1 $MASTER_ADDR"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

echo "=== Environment Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

echo "=== Starting Multi-Node Training (Direct srun) ==="

# 使用srun直接启动Python脚本，手动计算rank
srun bash -c "
    # 计算全局rank: RANK = SLURM_NODEID * GPUs_per_node + SLURM_LOCALID
    export RANK=\$((\$SLURM_NODEID * 2 + \$SLURM_LOCALID))
    export LOCAL_RANK=\$SLURM_LOCALID
    export WORLD_SIZE=$WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    echo \"[Node \$(hostname) - Global Rank \$RANK - Local Rank \$LOCAL_RANK] Starting training\"
    echo \"[Node \$(hostname)] MASTER_ADDR=\$MASTER_ADDR, MASTER_PORT=\$MASTER_PORT\"
    
    python train_debug.py
"

echo "Multi-node job completed." 