# conda create -n openvla python=3.10 -y

# conda activate openvla

# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# pip install -e .

# pip install packaging ninja

pip install "flash-attn==2.5.5" --no-build-isolation
