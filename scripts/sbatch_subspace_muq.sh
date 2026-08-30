#!/bin/bash
#SBATCH --job-name=topo-submuq
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/submuq_%A_%a.out
#SBATCH --error=logs/submuq_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
ARMS=(muq_L2 muq_L6 muq_L8 muq_L12)
# works from the cached clipkey and nsynth_emb npz files, so no model is loaded
.venv/bin/python scripts/13_tonal_subspace.py --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}" \
  --n-random 10
