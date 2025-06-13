# conda activate embodied

python vla-scripts/extern/convert_openvla_weights_to_hf.py \
    --openvla_model_path_or_id $1 \
    --output_hf_model_local_path $2