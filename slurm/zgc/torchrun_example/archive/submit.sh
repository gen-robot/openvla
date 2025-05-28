#!/bin/bash

#SBATCH --job-name=pytorch_torchrun_example    # 作业名称
#SBATCH --nodes=2                              # 请求3个节点
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=1                    # 每个节点只运行1个任务 (torchrun会处理多GPU)
#SBATCH --gpus-per-node=8                      # 每个节点请求8张GPU
#SBATCH --cpus-per-task=32                     # 每个任务分配32个CPU核心 (用于数据加载等)
#SBATCH --mem=64G                              # 每个节点内存 (可以根据需要调整)
#SBATCH --time=00:10:00                        # 作业运行时间限制 (小时:分钟:秒)
#SBATCH --output=logs/pytorch_torchrun_%j.out  # 标准输出文件，%j会被作业ID替换
#SBATCH --error=logs/pytorch_torchrun_%j.err   # 标准错误文件

# --- 环境设置 ---
# 加载你需要的模块 (例如 anaconda, cuda, cudnn, nccl)
# 这个部分根据你的集群环境具体配置
# module purge # 清理可能存在的模块冲突
# module load anaconda3/xxxx # 你的conda环境
# module load cuda/xx.x    # 你的cuda版本
# module load cudnn/x.x.x  # 你的cudnn版本
# module load nccl/x.x.x   # 你的nccl版本

# 激活你的conda环境 (如果使用conda)
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
# proxy_on
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# 如果需要更详细的日志，可以使用 TRACE，但 INFO 通常足够初步诊断
# export NCCL_DEBUG_SUBSYS=ALL # 可选，输出所有子系统的debug信息

export NCCL_SOCKET_IFNAME=bond0

# 确保你的环境中安装了 PyTorch, torchvision, nccl (通常PyTorch自带)

# --- MASTER Address 和 Port 配置 ---
# torchrun需要知道主节点地址和端口
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE_HOSTNAME
export MASTER_PORT=39501

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

# --- 使用 torchrun 启动分布式训练 ---
# torchrun 自动处理进程启动和环境变量设置
# --nnodes: 总节点数
# --nproc_per_node: 每个节点的进程数 (通常等于GPU数量)
# --node_rank: 当前节点的排名 (0 到 nnodes-1)
# --master_addr: 主节点地址
# --master_port: 主节点端口
# --rdzv_id: rendezvous ID，用于多节点同步 (可以使用作业ID)
# --rdzv_backend: rendezvous后端，c10d是默认选择
# --rdzv_endpoint: rendezvous端点，格式为 host:port

echo "Starting torchrun with the following parameters:"
echo "  --nnodes=$NNODES"
echo "  --nproc_per_node=$NPROC_PER_NODE"
echo "  --node_rank=$NODE_RANK"
echo "  --master_addr=$MASTER_ADDR"
echo "  --master_port=$MASTER_PORT"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py

echo "Job finished." 