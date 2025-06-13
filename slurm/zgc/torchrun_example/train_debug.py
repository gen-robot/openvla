import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import os
import functools
import time
import sys

# --- 配置参数 ---
NUM_EPOCHS = 3  # 减少epochs用于调试
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PRINT_EVERY_N_BATCHES = 10

def debug_print(msg, rank=None):
    """带时间戳的调试打印"""
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else "UNINIT"
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] [Rank {rank}] {msg}", flush=True)

# --- 分布式训练设置 ---
def setup_distributed():
    """初始化分布式训练环境 - 针对torchrun优化 - 调试版本"""
    debug_print("Starting distributed setup...")
    
    if not dist.is_available():
        debug_print("ERROR: torch.distributed not available!")
        return False
        
    if not dist.is_nccl_available():
        debug_print("ERROR: NCCL not available!")
        return False

    # 检查必要的环境变量
    debug_print("Checking environment variables...")
    env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
    for var in env_vars:
        value = os.environ.get(var, "NOT_SET")
        debug_print(f"  {var} = {value}")

    # torchrun 会自动设置这些环境变量
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    debug_print(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    debug_print(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    debug_print(f"Node hostname: {os.uname()[1]}")

    try:
        debug_print("Calling dist.init_process_group...")
        # 添加超时设置
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=torch.distributed.default_pg_timeout
        )
        debug_print("Process group initialized successfully!")
    except Exception as e:
        debug_print(f"ERROR initializing process group: {e}")
        import traceback
        traceback.print_exc()
        return False

    debug_print(f"Setting CUDA device to {local_rank}")
    torch.cuda.set_device(local_rank)
    debug_print(f"Rank {rank} is using GPU {local_rank} on node {os.uname()[1]}.")
    
    # 测试简单的all_reduce操作
    debug_print("Testing basic distributed communication...")
    try:
        test_tensor = torch.tensor([float(rank)]).cuda()
        dist.all_reduce(test_tensor)
        debug_print(f"All-reduce test successful! Result: {test_tensor.item()}")
    except Exception as e:
        debug_print(f"ERROR in all-reduce test: {e}")
        return False
    
    return True

def cleanup_distributed():
    """清理分布式训练环境"""
    rank = dist.get_rank() if dist.is_initialized() else "UNINIT"
    debug_print("Cleaning up distributed environment...", rank)
    if dist.is_initialized():
        dist.destroy_process_group()
    debug_print("Cleanup complete.", rank)

# --- 简化的测试模型 ---
class SimpleTestModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=10):
        super(SimpleTestModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

# --- 训练函数 ---
def train(rank, world_size, local_rank):
    debug_print(f"Starting training on rank {rank}/{world_size} (local GPU {local_rank}).")

    # 1. 创建模型和优化器
    debug_print("Creating model...")
    model = SimpleTestModel().to(local_rank)

    debug_print("Wrapping model with FSDP...")
    try:
        fsdp_model = FSDP(model, device_id=torch.cuda.current_device())
        debug_print("FSDP wrap successful!")
    except Exception as e:
        debug_print(f"ERROR wrapping model with FSDP: {e}")
        return

    optimizer = optim.Adam(fsdp_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 2. 准备数据 (简化的伪数据)
    debug_print("Creating dataset...")
    num_samples = 1000
    inputs = torch.randn(num_samples, 1024)
    labels = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(inputs, labels)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  # 设为0避免multiprocessing问题
        pin_memory=True
    )
    debug_print(f"DataLoader created with {len(dataloader)} batches per epoch")

    # 3. 训练循环
    debug_print("Starting training loop...")
    fsdp_model.train()
    for epoch in range(NUM_EPOCHS):
        debug_print(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")
        sampler.set_epoch(epoch)
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(local_rank), target.to(local_rank)

            optimizer.zero_grad()
            output = fsdp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and (batch_idx + 1) % PRINT_EVERY_N_BATCHES == 0:
                debug_print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # 同步所有进程
        debug_print("Synchronizing processes...")
        dist.barrier()
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            debug_print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    if rank == 0:
        debug_print("Training complete!")

# --- 主函数 ---
if __name__ == "__main__":
    debug_print("Python script started")
    debug_print(f"Python version: {sys.version}")
    debug_print(f"PyTorch version: {torch.__version__}")
    debug_print(f"CUDA available: {torch.cuda.is_available()}")
    debug_print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if not setup_distributed():
        debug_print("Failed to setup distributed environment. Exiting.")
        exit(1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    debug_print(f"Final distributed setup - rank={rank}, world_size={world_size}, local_rank={local_rank}")

    try:
        train(rank, world_size, local_rank)
    except Exception as e:
        debug_print(f"ERROR during training on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()
        debug_print(f"Rank {rank} finished and cleaned up.") 