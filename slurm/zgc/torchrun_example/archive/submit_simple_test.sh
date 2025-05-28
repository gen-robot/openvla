#!/bin/bash

#SBATCH --job-name=pytorch_simple_test         # 作业名称
#SBATCH --nodes=2                              # 请求2个节点
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=2                    # 每个节点2个任务 (对应2个GPU)
#SBATCH --gpus-per-node=2                      # 每个节点请求2张GPU
#SBATCH --cpus-per-task=4                      # 每个任务分配4个CPU核心
#SBATCH --mem=32G                              # 每个节点内存
#SBATCH --time=00:05:00                        # 作业运行时间限制
#SBATCH --output=logs/pytorch_simple_test_%j.out
#SBATCH --error=logs/pytorch_simple_test_%j.err

# --- 环境设置 ---
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=1

# --- 手动IP映射 ---
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

case $MASTER_NODE_HOSTNAME in
    "pm-9eea")
        export MASTER_ADDR="10.3.27.3"
        ;;
    "pm-9eea0001")
        export MASTER_ADDR="10.3.27.4"
        ;;
    *)
        export MASTER_ADDR=$MASTER_NODE_HOSTNAME
        ;;
esac

export MASTER_PORT=39511
export WORLD_SIZE=4  # 2 nodes × 2 GPUs = 4 total processes

echo "=== Simple Multi-Node Test ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# 网络测试
ping -c 1 $MASTER_ADDR && echo "✓ Ping successful" || echo "✗ Ping failed"

cd /data/home/fgao/openvla/slurm/zgc/torchrun_example/

echo "=== Starting Simple Distributed Test ==="

# 使用srun直接启动Python脚本，不使用torchrun
srun bash -c "
    # 计算全局rank
    export RANK=\$((\$SLURM_NODEID * 2 + \$SLURM_LOCALID))
    export LOCAL_RANK=\$SLURM_LOCALID
    export WORLD_SIZE=$WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    
    echo \"[Node \$(hostname) - Global Rank \$RANK - Local Rank \$LOCAL_RANK] Starting Python script\"
    echo \"[Node \$(hostname)] MASTER_ADDR=\$MASTER_ADDR, MASTER_PORT=\$MASTER_PORT\"
    
    python train_debug.py
"

echo "Simple test completed." 