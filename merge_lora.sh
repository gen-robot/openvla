base_path=$1
lora_path=$2

python vla-scripts/merge_lora_weights_and_save.py --base_checkpoint ${base_path} --lora_finetuned_checkpoint_dir ${lora_path}
