import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import os
import functools

# --- 配置参数 ---
# 这些参数也可以通过命令行参数传入
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PRINT_EVERY_N_BATCHES = 10

# --- 分布式训练设置 ---
def setup_distributed():
    """初始化分布式训练环境 - 针对torchrun优化"""
    if not dist.is_available() or not dist.is_nccl_available():
        print("Distributed training or NCCL backend not available.")
        return False

    # torchrun 会自动设置这些环境变量
    rank = int(os.environ.get("RANK", 0))  # 全局rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # 总进程数
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # 节点内rank
    
    # torchrun 还会设置以下变量:
    # MASTER_ADDR - 主节点地址
    # MASTER_PORT - 主节点端口
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    print(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    print(f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

    try:
        # torchrun 通常使用 env:// 初始化方法，它会自动从环境变量读取配置
        dist.init_process_group(
            backend="nccl",  # 推荐使用 NCCL 进行 GPU 训练
            init_method="env://",  # 使用环境变量初始化
        )
    except Exception as e:
        print(f"Error initializing process group: {e}")
        print("Check MASTER_ADDR, MASTER_PORT, network connectivity, and NCCL setup.")
        return False

    torch.cuda.set_device(local_rank)  # 将当前进程绑定到特定的 GPU
    print(f"Rank {rank} is using GPU {local_rank} on node {os.uname()[1]}.")
    return True

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

# --- 定义简单模型 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes) # 假设输入是 28x28

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

# --- 训练函数 ---
def train(rank, world_size, local_rank):
    print(f"Starting training on rank {rank}/{world_size} (local GPU {local_rank}).")

    # 1. 创建模型和优化器
    model = SimpleCNN().to(local_rank) # 模型先放到对应的 GPU 上

    # FSDP 自动包装策略 (可以根据模型层的大小来决定是否包装)
    # my_auto_wrap_policy = functools.partial(
    #     size_based_auto_wrap_policy, min_num_params=1000 # 示例：参数量大于1000的层会被单独FSDP包装
    # )
    # fsdp_model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy, device_id=torch.cuda.current_device())
    fsdp_model = FSDP(model, device_id=torch.cuda.current_device()) # 简单起见，直接包装整个模型

    optimizer = optim.Adam(fsdp_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 2. 准备数据 (伪数据)
    # 在实际应用中，你会从磁盘加载数据
    # 确保每个 rank 只加载其数据的一部分，或者使用 DistributedSampler
    num_samples = 1024 * world_size # 总样本数
    inputs = torch.randn(num_samples, 1, 28, 28) # (N, C, H, W)
    labels = torch.randint(0, 10, (num_samples,))

    # 使用 DistributedSampler 来确保每个 GPU 得到不同的数据子集
    dataset = TensorDataset(inputs, labels)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True # 通常在训练时打乱
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2, # 根据你的机器配置调整
        pin_memory=True
    )

    # 3. 训练循环
    fsdp_model.train()
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch) # 对于 DistributedSampler，需要在每个 epoch 开始前调用
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
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # 在所有进程上同步，并只在 rank 0 打印平均 loss (可选)
        dist.barrier() # 等待所有进程完成当前 epoch
        avg_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f} on all ranks.")

    if rank == 0:
        print("Training complete!")
        # 你可以在这里保存模型 (注意FSDP保存和加载的方式)
        # dist_state_dict = fsdp_model.state_dict() # 获取分布式状态字典
        # if rank == 0:
        # torch.save(dist_state_dict, "fsdp_model.pt")
        # print("Model saved as fsdp_model.pt")

# --- 主函数 ---
if __name__ == "__main__":
    if not setup_distributed():
        print("Failed to setup distributed environment. Exiting.")
        exit(1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["SLURM_LOCALID"]) # 或者 rank % torch.cuda.device_count()

    try:
        train(rank, world_size, local_rank)
    except Exception as e:
        print(f"Error during training on rank {rank}: {e}")
    finally:
        cleanup_distributed()
        print(f"Rank {rank} finished and cleaned up.")