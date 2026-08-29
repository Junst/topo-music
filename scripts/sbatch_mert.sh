#!/bin/bash
#SBATCH --job-name=topo-mert
# All three RTX3090 partitions: the nodes have CPUs and memory free but
# every 3090 is allocated, so widening the pool is what shortens the wait.
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/mert_%A_%a.out
#SBATCH --error=logs/mert_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
LAYERS=(0 4 8 12 16 20 24)
L=${LAYERS[$SLURM_ARRAY_TASK_ID]}
echo "layer=$L host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/05_feature_codebook.py --feat mert --layer "$L" \
  --k 1024 --batch 16 --device cuda
.venv/bin/python scripts/04_semantic_topography.py --probe "runs/probe_mert_L$L.npz" --n-perm 200
echo "done=$(date -Is)"
