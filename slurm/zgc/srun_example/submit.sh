#!/bin/bash

#SBATCH --job-name=pytorch_distributed_example # 作业名称
#SBATCH --nodes=3                              # 请求2个节点
#SBATCH --exclude=pm-9eea0003                  # 排除节点pm-9eea0003
#SBATCH --ntasks-per-node=8                    # 每个节点运行2个任务 (假设每个节点有2张GPU)
#SBATCH --gpus-per-node=8                      # 每个节点请求2张GPU
#SBATCH --cpus-per-task=4                      # 每个任务分配4个CPU核心 (用于数据加载等)
#SBATCH --mem=16G                              # 每个节点内存 (可以根据需要调整)
#SBATCH --time=00:10:00                        # 作业运行时间限制 (小时:分钟:秒)
#SBATCH --output=logs/pytorch_distributed_%j.out    # 标准输出文件，%j会被作业ID替换
#SBATCH --error=logs/pytorch_distributed_%j.err     # 标准错误文件

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
export MASTER_NODE_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# 尝试获取主节点的IB IP (如果ssh不通或接口名不确定，这里会回退到使用主机名)
# 为了简化，暂时依赖 NCCL_SOCKET_IFNAME。如果主机名解析和路由配置良好，这可能也行。
# MASTER_IB_IP=$(ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null $MASTER_NODE_HOSTNAME "ip -4 addr show mlx5_ib1 | grep -oP 'inet \K[\d.]+'" 2>/dev/null)
# export MASTER_ADDR=${MASTER_IB_IP:-$MASTER_NODE_HOSTNAME}
export MASTER_ADDR=$MASTER_NODE_HOSTNAME # 先尝试用主机名，配合正确的NCCL_SOCKET_IFNAME


export MASTER_PORT=39501 

echo "Master Node Hostname: $MASTER_NODE_HOSTNAME"
echo "Attempting MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $SLURM_NTASKS"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
# --- 计算 world_size ---
# SLURM_NTASKS 已经包含了总的任务数 (world_size)
export WORLD_SIZE=$SLURM_NTASKS
echo "WORLD_SIZE: $WORLD_SIZE"

# --- 获取每个节点上的任务数，用于计算local_rank ---
# 这个例子中，我们让 SLURM_PROCID (全局rank) 直接作为PyTorch的rank
# PyTorch脚本内部会通过 rank % gpus_per_node 来确定local_gpu_id

# --- 运行PyTorch训练脚本 ---
# 使用srun启动所有任务
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE" # 这个可能没有，但 gpus-per-task 或 gpus-per-node 会设置
echo "SLURM_PROCID: $SLURM_PROCID" # srun 会设置这个
echo "SLURM_LOCALID: $SLURM_LOCALID" # srun 会设置这个
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" # srun 通常会设置

# --- 获取主节点地址和端口 ---
# scontrol show hostnames $SLURM_JOB_NODELIST 可以获取节点列表

# 选择一个未被占用的端口，或者让 PyTorch 自动选择 (但指定更可靠)

export WORLD_SIZE=$SLURM_NTASKS # 总的进程数 (等于总的 GPU 数量)

echo "WORLD_SIZE: $WORLD_SIZE"

cd /data/home/fgao/openvla/slurm/zgc/example/

# --- 执行训练 ---
# 使用 srun 来启动分布式任务
# --export=ALL 确保 Slurm 的环境变量被传递给 python 进程
# --mpi=pmix 或者 --mpi=pmi2 是一些集群需要的，用于 srun 的分布式启动
# 如果你的集群不需要显式指定 MPI 类型，可以去掉 --mpi 选项
# 如果你的 Python 脚本直接从 os.environ 读取 SLURM_* 变量，则可能不需要 --export=ALL
# 但是为了保险起见，加上通常是好的。

# `srun` 会为每个任务设置 SLURM_PROCID (全局 rank) 和 SLURM_LOCALID (节点内 rank)
# Python 脚本会使用这些变量

# 每个节点有多少个 GPU，`--ntasks-per-node` 就应该等于这个数量
# `--gpus-per-task=1` 表示每个 srun 任务分配一个 GPU
# `srun` 将会在每个分配到的节点上，为 `--ntasks-per-node` 指定数量的任务（每个任务一个GPU）启动 `python fsdp_train.py`
echo "Value of MASTER_PORT in sbatch script (before srun): $MASTER_PORT"
srun --export=ALL python train.py

echo "Job finished."