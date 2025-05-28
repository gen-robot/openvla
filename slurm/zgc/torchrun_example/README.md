# PyTorch Distributed Training Examples for SLURM

This directory contains clean, tested examples for running distributed PyTorch training on SLURM clusters using both `srun` and `torchrun`.

## 📁 Directory Structure

```
torchrun_example/
├── 1_single_node_test.sh      # ✅ Single-node 8 GPUs (WORKING)
├── 2_multi_node_srun.sh       # 🧪 Multi-node using srun (TO TEST)
├── 3_multi_node_torchrun.sh   # 🧪 Multi-node using torchrun (TO TEST)
├── train_debug.py             # Training script with debug output
├── test_finetune_setup.py     # VLA-style test script
├── logs/                      # Job output logs
├── archive/                   # Old experimental scripts
└── README.md                  # This file
```

## 🚀 Step-by-Step Testing Plan

### Step 1: Single-Node Test (VERIFIED WORKING ✅)
```bash
sbatch 1_single_node_test.sh
```

**What it does:**
- Tests 8 GPUs on a single node
- Uses `torchrun` with `localhost`
- Validates FSDP, NCCL, and distributed communication
- **Status: CONFIRMED WORKING** ✅

**Expected output:**
- All 8 ranks initialize successfully
- NCCL all-reduce test passes
- Training loss decreases over epochs
- Clean shutdown with no errors

---

### Step 2: Multi-Node with srun (RECOMMENDED NEXT TEST 🧪)
```bash
sbatch 2_multi_node_srun.sh
```

**What it does:**
- Tests 2 nodes × 2 GPUs = 4 total processes
- Uses direct `srun` (no torchrun)
- Manually calculates ranks: `RANK = SLURM_NODEID * 2 + SLURM_LOCALID`
- **Status: TO BE TESTED** 🧪

**Why test this first:**
- Simpler than torchrun (no rendezvous complexity)
- Direct control over rank assignment
- Easier to debug network issues
- Similar to your working srun example

---

### Step 3: Multi-Node with torchrun (ADVANCED TEST 🧪)
```bash
sbatch 3_multi_node_torchrun.sh
```

**What it does:**
- Tests 2 nodes × 2 GPUs = 4 total processes
- Uses `torchrun` with proper rendezvous
- Automatic rank assignment by torchrun
- **Status: TO BE TESTED** 🧪

**Why test this last:**
- More complex (torchrun rendezvous)
- Requires all nodes to start simultaneously
- Validates torchrun for your finetune.py usage

---

## 🔧 Key Technical Details

### Environment Variables
- **srun approach**: Manual `RANK`, `LOCAL_RANK`, `WORLD_SIZE`
- **torchrun approach**: Automatic rank assignment

### Network Configuration
- **MASTER_ADDR**: Manually mapped IPs (pm-9eea → 10.3.27.3, etc.)
- **NCCL**: Uses `bond0` interface with InfiniBand settings
- **Ports**: Different for each test (39501, 39512, 39513)

### Resource Allocation
- **Single-node**: 1 node × 8 GPUs × 32 CPUs
- **Multi-node**: 2 nodes × 2 GPUs × 8 CPUs (reduced for testing)

---

## 📊 Expected Results

### Success Indicators
1. **Environment setup**: PyTorch, CUDA, NCCL all available
2. **Network connectivity**: Ping tests pass between nodes
3. **Distributed init**: All ranks initialize without timeout
4. **Communication test**: NCCL all-reduce returns correct result
5. **Training**: Loss decreases over epochs
6. **Clean shutdown**: No hanging processes

### Common Issues & Solutions
- **Timeout errors**: Check network connectivity and IP mapping
- **Hanging processes**: Ensure all nodes start simultaneously
- **NCCL errors**: Verify InfiniBand configuration
- **Port conflicts**: Each test uses different ports

---

## 🎯 Validation for finetune.py

These tests validate that torchrun will work with your actual `finetune.py` because:

1. **Same distributed patterns**: FSDP, gradient accumulation, multi-GPU
2. **Same environment**: PyTorch 2.2.0+cu121, NCCL, InfiniBand
3. **Same SLURM setup**: Resource allocation, node exclusion
4. **Same network config**: bond0 interface, IP mapping

Once these tests pass, you can confidently use the same torchrun setup for your VLA fine-tuning.

---

## 📝 Next Steps After Testing

1. **If Step 2 (srun) works**: You have a reliable fallback method
2. **If Step 3 (torchrun) works**: You can use torchrun for finetune.py
3. **Scale up**: Test with more nodes/GPUs once basic setup works
4. **Production**: Apply the working pattern to your actual training scripts

---

## 🗂️ Archive

The `archive/` folder contains previous experimental scripts that had various issues:
- `submit_fixed.sh` - IP resolution problems
- `submit_manual_ip.sh` - Torchrun hanging issues
- `submit_debug.sh` - Early debugging version

These are kept for reference but should not be used for testing.

---

## 🔧 Key Differences: srun vs torchrun

### Resource Allocation
**srun approach:**
```bash
#SBATCH --ntasks-per-node=8    # One task per GPU
#SBATCH --cpus-per-task=4      # CPUs per GPU
```

**torchrun approach:**
```bash
#SBATCH --ntasks-per-node=1    # One task per node
#SBATCH --cpus-per-task=32     # All CPUs shared among GPUs
```

### Launch Commands
**srun:**
```bash
srun --export=ALL python train.py
```

**torchrun:**
```bash
torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
         --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT train.py
```

### Advantages
**torchrun benefits:**
- Better error handling and fault tolerance
- Elastic training support
- Simplified configuration
- Standard PyTorch environment variables

**srun benefits:**
- Tighter SLURM integration
- Fine-grained resource control
- Works with any distributed framework 