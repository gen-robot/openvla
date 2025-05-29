
export SIF_IMAGE="/storage/openpsi/images/mlaas-eai-v2.6.sif"

srun --mpi=pmi2 \
    --gpus=8 \
    --pty \
    singularity exec --nv \
    --pid \
    --writable-tmpfs \
    --no-home \
    --bind /storage:/storage \
    --env WANDB_BASE_URL=http://8.150.1.98:8080 \
    --env WANDB_API_KEY=local-862bc023bdff4309bad1e6cc319369cdb354b4ac \
    $SIF_IMAGE \
    bash -c "

    echo $http_proxy $https_proxy $all_proxy
    export http_proxy=; export https_proxy=; export HTTP_PROXY=; export HTTPS_PROXY=; export all_proxy=; export ALL_PROXY=
    echo $http_proxy $https_proxy $all_proxy
    
    # Setup directories
    mkdir -p /code/EmbodiedAgent/embodied_agent/third_party/vla/
    cd /code/EmbodiedAgent/embodied_agent/third_party/vla/
    ln -sf /storage/openpsi/users/gaofeng/openvla

    cd /storage/openpsi/users/gaofeng/openvla/
    . '/opt/conda/etc/profile.d/conda.sh'
    conda activate embodied

    bash"