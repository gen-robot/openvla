torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py  \
  --vla.type "prism-dinosiglip-224px+mx-bridge"  \
  --vla.data_mix libero_object_no_noops \
  --data_root_dir datasets/libero_data  \
  --run_root_dir runs  \
  --wandb_project VLA-Reasoning-debug \
  --enable_cot True