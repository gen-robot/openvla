#!/bin/bash

#SBATCH --job-name=vla_finetune_single             # 作业名称
#SBATCH --nodes=1                                  # 单节点
#SBATCH --ntasks-per-node=1                        # 每个节点1个任务 (torchrun处理多GPU)
#SBATCH --gpus-per-node=8                          # 每个节点8张GPU
#SBATCH --cpus-per-task=32                         # 每个任务32个CPU核心
#SBATCH --mem=256G                                 # 节点内存 (VLA需要更多内存)
#SBATCH --time=24:00:00                            # 运行时间限制 (24小时)
#SBATCH --output=logs/vla_finetune_single_%j.out
#SBATCH --error=logs/vla_finetune_single_%j.err

# --- 环境设置 ---
. $HOME/.bashrc
source /data/home/fgao/miniconda3/etc/profile.d/conda.sh
conda activate openvla

# 设置NCCL参数 (与测试中验证的相同)
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_5:1,mlx5_8:1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=1

# --- VLA Fine-tuning 配置 (基于原始 finetune.sh) ---
task_name="libero_object_no_noops"
use_film=False
use_proprio=False
num_images_in_input=1
use_l1_regression=False
use_diffusion=False
merge_lora_during_training=False
num_actions_chunk=1
use_parallel_decoding=False
is_debug=False
enable_cot=False
use_lora=True
cot_full=True
resume=False
vla_path=openvla/openvla-7b
cot_tags="move_reason,move"

# 单节点配置
num_gpus=8
project_name="VLA-Reasoning"

# --- 单节点torchrun配置 ---
export MASTER_ADDR=localhost
export MASTER_PORT=39521
export NNODES=1
export NPROC_PER_NODE=$num_gpus

echo "=== VLA Fine-tuning Setup (Single Node) ==="
echo "Task: $task_name"
echo "VLA Path: $vla_path"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NNODES: $NNODES"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "Current node: $(hostname)"

cd /data/home/fgao/openvla

echo "=== Environment Verification ==="
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "NCCL available: $(python -c 'import torch; print(torch.distributed.is_nccl_available())')"

# 创建日志目录
mkdir -p logs

echo "=== Starting VLA Fine-tuning ==="

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    vla-scripts/finetune.py \
    --vla_path ${vla_path} \
    --data_root_dir datasets/libero_data \
    --dataset_name ${task_name} \
    --run_root_dir checkpoints/${task_name} \
    --use_proprio ${use_proprio} \
    --use_film ${use_film} \
    --num_images_in_input ${num_images_in_input} \
    --use_lora ${use_lora} \
    --lora_rank 32 \
    --batch_size 4 \
    --grad_accumulation_steps 2 \
    --learning_rate 5e-4 \
    --image_aug False \
    --wandb_project ${project_name} \
    --max_steps 100_000 \
    --merge_lora_during_training ${merge_lora_during_training} \
    --use_l1_regression ${use_l1_regression} \
    --use_diffusion ${use_diffusion} \
    --use_parallel_decoding ${use_parallel_decoding} \
    --enable_cot ${enable_cot} \
    --num_actions_chunk ${num_actions_chunk} \
    --use_val_set True \
    --save_freq 5000 \
    --val_freq 1000

echo "VLA fine-tuning completed." 