#!/bin/bash
#SBATCH --job-name=topo-e2
# All three RTX3090 partitions: the nodes have CPUs and memory free but
# every 3090 is allocated, so widening the pool is what shortens the wait.
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/e2_%A_%a.out
#SBATCH --error=logs/e2_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
echo "task=$SLURM_ARRAY_TASK_ID host=$(hostname) start=$(date -Is)"
case $SLURM_ARRAY_TASK_ID in
  0) .venv/bin/python scripts/03_probe_nsynth.py --codec encodec_32k --batch 32 --device cuda
     .venv/bin/python scripts/04_semantic_topography.py --probe runs/probe_encodec_32k.npz --n-perm 200 ;;
  1) .venv/bin/python scripts/03_probe_nsynth.py --codec encodec_24k --batch 32 --device cuda
     .venv/bin/python scripts/04_semantic_topography.py --probe runs/probe_encodec_24k.npz --n-perm 200 ;;
  2) .venv/bin/python scripts/03_probe_nsynth.py --codec dac_44k --batch 32 --device cuda
     .venv/bin/python scripts/04_semantic_topography.py --probe runs/probe_dac_44k.npz --n-perm 200 ;;
  3) .venv/bin/python scripts/05_feature_codebook.py --feat cqt --k 1024 --batch 16 --device cpu
     .venv/bin/python scripts/04_semantic_topography.py --probe runs/probe_cqt.npz --n-perm 200 ;;
esac
echo "done=$(date -Is)"
