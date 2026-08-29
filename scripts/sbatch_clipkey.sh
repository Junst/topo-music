#!/bin/bash
#SBATCH --job-name=topo-ck
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/ck_%A_%a.out
#SBATCH --error=logs/ck_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
ARMS=(cqt mert_L4 mert_L12 mert_L16 mert_L24 encodec_32k)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/09_clip_key_geometry.py --arm "$A" --seconds 20 \
  --batch 8 --n-perm 2000 --device cuda
echo "done=$(date -Is)"
