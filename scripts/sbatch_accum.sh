#!/bin/bash
#SBATCH --job-name=topo-acc
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/acc_%A_%a.out
#SBATCH --error=logs/acc_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
ARMS=(mert_L12 chroma cqt mert_L4 mert_L16 mert_L24)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
DEV=cuda; [[ "$A" == chroma || "$A" == cqt ]] && DEV=cpu
.venv/bin/python scripts/14_temporal_accumulation.py --arm "$A" --seconds 20 \
  --batch 8 --repeats 5 --n-perm 1000 --device $DEV
