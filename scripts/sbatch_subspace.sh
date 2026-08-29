#!/bin/bash
#SBATCH --job-name=topo-sub
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/sub_%A_%a.out
#SBATCH --error=logs/sub_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
ARMS=(mert_L12 mert_L4 mert_L16 mert_L24 chroma cqt encodec_32k)
.venv/bin/python scripts/13_tonal_subspace.py --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}" \
  --n-random 10
