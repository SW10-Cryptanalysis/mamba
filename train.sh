#!/bin/bash
set -eo pipefail

# 1. Navigate to your mounted workspace
cd /work

# 2. Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "mamba" ]; then
    echo "Cloning repository..."
    git clone https://github.com/SW10-Cryptanalysis/mamba.git
    cd mamba
else
    echo "Git pulling newest changes..."
    cd mamba
    git pull
fi

mkdir -p logs
export HF_HUB_ENABLE_HF_TRANSFER=1
export UV_CACHE_DIR="/work/.uv_cache"

# 3. Dynamically count available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPU(s)..."

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

export PYTORCH_ALLOC_CONF="expandable_segments:True"

# 4. L40 Setup & NCCL Optimizations
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [ "$NUM_GPUS" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_P2P_LEVEL=SYS
    export NCCL_NET_GDR_LEVEL=SYS
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv --python 3.12
fi

# Install project dependencies
uv pip install -e ".[gpu]"

# Install hf_transfer to enable faster Hugging Face downloads
uv pip install hf_transfer

MASTER_PORT=$((10000 + $RANDOM % 20000))

# 5. Launch Training
echo "Launching torchrun with $NUM_GPUS processes..."
uv run torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m src.train "$@"

echo "Training Job finished at $(date)"****