#!/bin/bash
#SBATCH --job-name=agentserve-validate
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00
#SBATCH --output=agentserve-%j.out
#SBATCH --error=agentserve-%j.err

# Validation run for UB CCR (or any Slurm cluster with GPU nodes).
#
#   sbatch scripts/ccr_validate.slurm
#   squeue -u $USER
#   tail -f agentserve-<jobid>.out
#
# Adjust --partition/--qos to whatever your allocation grants; check with
# `slimits` on CCR. Ask for the smallest GPU that fits -- a V100 or A100 is
# plenty, and small requests queue far faster than large ones.
#
# Everything runs on the compute node's loopback interface, so no ports are
# exposed off-node and no firewall rules are needed.

set -euo pipefail

module load python || true
module load cuda   || true

echo "node      : $(hostname)"
echo "gpus      :"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Node-local scratch keeps the model download and venv off shared filesystems,
# which is both faster and politer to other users.
SCRATCH="${SLURMTMPDIR:-/tmp}/agentserve-$SLURM_JOB_ID"
mkdir -p "$SCRATCH"
export HF_HOME="$SCRATCH/hf"
export VLLM_CACHE_ROOT="$SCRATCH/vllm"

python -m venv "$SCRATCH/venv"
source "$SCRATCH/venv/bin/activate"
pip install -q --upgrade pip
pip install -q "vllm>=0.6.0"
pip install -q -e .

echo "--- unit tests (fast, no GPU) ---"
python -m pytest -q

echo "--- GPU validation ---"
python scripts/gpu_validate.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --replicas 2 \
    --sessions 48 --turns 12 --concurrency 16 \
    --prompt-tokens 2000 \
    --kv-budget-tokens 60000 \
    --max-model-len 16384 \
    --out "$SLURM_SUBMIT_DIR/gpu_validation-$SLURM_JOB_ID.json"

echo "results written to gpu_validation-$SLURM_JOB_ID.json"
