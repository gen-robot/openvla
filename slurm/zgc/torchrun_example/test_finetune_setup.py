import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import os
import functools
from accelerate import PartialState

# --- 配置参数 (模拟finetune.py的参数) ---
NUM_EPOCHS = 5  # 减少epochs用于测试
BATCH_SIZE = 4  # 小batch size用于测试
LEARNING_RATE = 5e-4
PRINT_EVERY_N_BATCHES = 5
GRAD_ACCUMULATION_STEPS = 2  # 模拟finetune.py的梯度累积

# --- 分布式训练设置 (模拟finetune.py的setup) ---
def setup_distributed_like_finetune():
    """初始化分布式训练环境 - 模拟finetune.py的设置"""
    if not dist.is_available() or not dist.is_nccl_available():
        print("Distributed training or NCCL backend not available.")
        return False, None

    # 使用accelerate的PartialState (就像finetune.py一样)
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # 获取分布式训练参数 (模拟finetune.py的方式)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Distributed training configuration:")
    print(f"\tTotal processes (WORLD_SIZE): {world_size}")
    print(f"\tProcesses per node (LOCAL_WORLD_SIZE): {local_world_size}")
    print(f"\tGlobal rank: {rank}")
    print(f"\tLocal rank: {local_rank}")
    print(f"\tDevice ID: {device_id}")
    print(f"\tNode hostname: {os.uname()[1]}")
    
    # 计算有效batch size (就像finetune.py一样)
    effective_batch_size = BATCH_SIZE * GRAD_ACCUMULATION_STEPS * world_size
    print(f"\tEffective batch size: {effective_batch_size}")

    return True, distributed_state

# --- 模拟VLA模型的复杂模型 ---
class MockVLAModel(nn.Module):
    """模拟OpenVLA模型的结构，有更复杂的层次用于测试FSDP"""
    def __init__(self, vocab_size=1000, hidden_dim=512, num_layers=8):
        super(MockVLAModel, self).__init__()
        
        # 模拟vision backbone
        self.vision_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim)
        )
        
        # 模拟language model layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 模拟action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 7)  # 7-DOF action
        )
        
    def forward(self, pixel_values, input_ids):
        # 模拟视觉特征提取
        vision_features = self.vision_backbone(pixel_values)  # (B, hidden_dim)
        
        # 模拟语言特征
        text_features = self.embedding(input_ids)  # (B, seq_len, hidden_dim)
        
        # 简单融合 (实际VLA会更复杂)
        batch_size, seq_len, _ = text_features.shape
        vision_features_expanded = vision_features.unsqueeze(1).expand(-1, seq_len, -1)
        fused_features = text_features + vision_features_expanded
        
        # 通过transformer layers
        for layer in self.transformer_layers:
            fused_features = layer(fused_features)
        
        # 预测动作 (使用最后一个token的特征)
        actions = self.action_head(fused_features[:, -1])  # (B, 7)
        
        return actions

def wrap_model_with_fsdp(model, device_id):
    """使用FSDP包装模型，模拟finetune.py的方式"""
    # 定义FSDP包装策略 (类似finetune.py)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, 
        min_num_params=1000  # 参数量大于1000的层会被单独包装
    )
    
    # 使用FSDP包装 (模拟finetune.py的设置)
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device_id,
        # mixed_precision=...,  # 可以添加混合精度
    )
    
    return fsdp_model

def create_mock_dataloader(world_size, rank, device_id):
    """创建模拟数据加载器，模拟finetune.py的数据格式"""
    # 模拟VLA训练数据
    num_samples = 1000
    seq_len = 32
    
    # 模拟图像数据 (RGB, 224x224)
    pixel_values = torch.randn(num_samples, 3, 224, 224)
    
    # 模拟token序列
    input_ids = torch.randint(0, 1000, (num_samples, seq_len))
    
    # 模拟动作标签
    actions = torch.randn(num_samples, 7)  # 7-DOF actions
    
    dataset = TensorDataset(pixel_values, input_ids, actions)
    
    # 使用DistributedSampler (就像finetune.py)
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
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader

# --- 训练函数 (模拟finetune.py的训练循环) ---
def train_like_finetune(distributed_state):
    """模拟finetune.py的训练过程"""
    device_id = distributed_state.local_process_index
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Starting training on rank {rank}/{world_size} (device {device_id}).")

    # 1. 创建并包装模型 (模拟finetune.py)
    model = MockVLAModel().to(device_id)
    model = wrap_model_with_fsdp(model, device_id)
    
    # 统计参数数量 (模拟finetune.py)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # 2. 创建优化器 (模拟finetune.py)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 简单的L1/L2损失用于动作预测

    # 3. 创建数据加载器
    dataloader = create_mock_dataloader(world_size, rank, device_id)

    # 4. 训练循环 (模拟finetune.py的梯度累积)
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(NUM_EPOCHS):
        dataloader.sampler.set_epoch(epoch)  # 重要：为DistributedSampler设置epoch
        total_loss = 0.0
        
        for batch_idx, (pixel_values, input_ids, target_actions) in enumerate(dataloader):
            # 移动数据到设备
            pixel_values = pixel_values.to(device_id).to(torch.float32)
            input_ids = input_ids.to(device_id)
            target_actions = target_actions.to(device_id).to(torch.float32)

            # 前向传播
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predicted_actions = model(pixel_values, input_ids)
                loss = criterion(predicted_actions, target_actions)

            # 梯度累积 (模拟finetune.py)
            normalized_loss = loss / GRAD_ACCUMULATION_STEPS
            normalized_loss.backward()

            total_loss += loss.item()

            # 每GRAD_ACCUMULATION_STEPS步骤后更新参数
            if (batch_idx + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                gradient_step = (batch_idx + 1) // GRAD_ACCUMULATION_STEPS
                if rank == 0 and gradient_step % PRINT_EVERY_N_BATCHES == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{gradient_step}], Avg Loss: {avg_loss:.4f}")

        # 同步所有进程 (模拟finetune.py)
        dist.barrier()
        avg_epoch_loss = total_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

    if rank == 0:
        print("Training complete! This validates the distributed setup for finetune.py")

# --- 主函数 ---
if __name__ == "__main__":
    success, distributed_state = setup_distributed_like_finetune()
    if not success:
        print("Failed to setup distributed environment. Exiting.")
        exit(1)

    try:
        train_like_finetune(distributed_state)
    except Exception as e:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"Error during training on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        print(f"Rank {dist.get_rank() if dist.is_initialized() else 0} finished and cleaned up.") 