#!/bin/bash
#SBATCH --job-name=nosp-mamba2-cipher-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
# Navigate to project root
cd /ceph/project/SW10-CausalLM/mamba
mkdir -p logs
mkdir -p .triton_cache
mkdir -p .tmp 
# Define paths for clarity
CONTAINER="./pytorch_25.03-py3.sif"
VENV_DIR="./mamba-2-pt2503"
MASTER_PORT=$((10000 + $RANDOM % 20000))
echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date)"
# --- RUN TRAINING INSIDE SINGULARITY ---
# We use 'singularity exec' to run the entire shell block
# This ensures the venv activation and uv run happen INSIDE the container
# Inside your run_training.sh, replace the singularity block with this:
# Define the host library path
HOST_LIB_PATH="/usr/lib/x86_64-linux-gnu"

TRAIN_ARGS="$@"

singularity exec --nv \
    --pwd /ceph/project/SW10-CausalLM/mamba \
    -B /ceph/project/SW10-CausalLM:/ceph/project/SW10-CausalLM \
    -B ${VENV_DIR}:/scratch/venv \
    ${CONTAINER} /bin/bash -c "
        source /scratch/venv/bin/activate

	export PYTHONPATH=/scratch/venv/lib/python3.12/site-packages:\$PYTHONPATH
        # This helps the compiler find necessary headers
        export CPATH=\"/usr/local/cuda/include:\$CPATH\"

        export TRITON_CACHE_DIR=\"/ceph/project/SW10-CausalLM/mamba/.triton_cache\"
        export TMPDIR=\"/ceph/project/SW10-CausalLM/mamba/.tmp\"

        # This tells the system where to find the library at runtime
        export LD_LIBRARY_PATH=\"/.singularity.d/libs:\$LD_LIBRARY_PATH\"
        export OMP_NUM_THREADS=8
        torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
            -m src.train $TRAIN_ARGS 2>&1
" | tee -a logs/train_live_${SLURM_JOB_ID}.log
