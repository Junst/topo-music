#!/bin/bash
#SBATCH --job-name=topo-fold
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/fold_%A_%a.out
#SBATCH --error=logs/fold_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
ARMS=(cqt_fold cqt_norm)
.venv/bin/python scripts/09_clip_key_geometry.py --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}" \
  --seconds 20 --batch 8 --n-perm 2000 --device cpu
