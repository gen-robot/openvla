# 🧪 SLURM Testing Plan - Quick Reference

## Current Status
- ✅ **Step 1 COMPLETED**: Single-node 8 GPUs working perfectly
- 🧪 **Step 2 NEXT**: Multi-node with srun (recommended)
- 🧪 **Step 3 LATER**: Multi-node with torchrun (advanced)

## Commands to Run

### Step 1: Single-Node Test ✅ DONE
```bash
sbatch 1_single_node_test.sh
# Status: CONFIRMED WORKING
```

### Step 2: Multi-Node srun 🧪 NEXT
```bash
sbatch 2_multi_node_srun.sh
# Expected: Should work (similar to your working srun example)
# Resources: 2 nodes × 2 GPUs = 4 total processes
```

### Step 3: Multi-Node torchrun 🧪 LATER
```bash
sbatch 3_multi_node_torchrun.sh
# Expected: More complex, test after Step 2 works
# Resources: 2 nodes × 2 GPUs = 4 total processes
```

## Check Results
```bash
# Check job status
squeue -u $USER

# Check logs (replace JOB_ID with actual job ID)
tail -f logs/multi_node_srun_JOB_ID.out
tail -f logs/multi_node_srun_JOB_ID.err
```

## Success Criteria
1. **Network**: Ping tests pass between nodes
2. **Distributed**: All ranks initialize successfully  
3. **Communication**: NCCL all-reduce test passes
4. **Training**: Loss decreases over epochs
5. **Shutdown**: Clean exit, no hanging processes

## If Step 2 Works
- You have a reliable multi-node setup
- Can proceed to test torchrun (Step 3)
- Can apply same pattern to finetune.py

## If Step 2 Fails
- Check network connectivity issues
- Verify IP mapping in the script
- Debug with single-node first (Step 1)

---
**Goal**: Validate distributed setup for your VLA finetune.py usage 