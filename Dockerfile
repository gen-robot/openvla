# 使用 Miniconda 作为基础镜像
FROM continuumio/miniconda3

# 设定工作目录
WORKDIR /workspace

# 创建 Conda 环境并安装 Python 3.10
RUN conda create -n libero-openvla python=3.10 -y && \
    echo "conda activate libero-openvla" >> ~/.bashrc

# 切换到 Conda 环境，并安装 PyTorch 及 CUDA 相关库
RUN conda run -n libero-openvla conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 复制当前目录代码到容器内
COPY . /workspace/

# 切换到 Conda 环境，安装 Python 依赖
RUN conda run -n libero-openvla pip install -e . && \
    conda run -n libero-openvla pip install packaging ninja && \
    conda run -n libero-openvla pip install "flash-attn==2.5.5" --no-build-isolation

# 克隆 LIBERO 并安装
RUN mkdir -p /workspace/external && cd /workspace/external && \
    git clone https://github.com/gen-robot/LIBERO.git && \
    cd LIBERO && \
    conda run -n libero-openvla pip install -e .

# 额外安装所需 Python 库
RUN conda run -n libero-openvla pip install robosuite==1.4.1 bddl==3.5.0 easydict==1.13 cloudpickle==3.1.1 gym==0.26.2 imageio-ffmpeg==0.6.0

# 设定默认 Shell 以支持 Conda
SHELL ["conda", "run", "-n", "libero-openvla", "/bin/bash", "-c"]

# 设定默认启动命令
CMD ["/bin/bash"]
