#!/bin/bash
#SBATCH --job-name=topo-gs
# All three RTX3090 partitions: the a100 queue is occupied by the user's own
# EQ-JEPA screen jobs and this work does not need an a100.
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/gs_%A_%a.out
#SBATCH --error=logs/gs_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
ARMS=(cqt mert_L4 mert_L16 mert_L24 encodec_32k)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/06_probe_gs.py --arm "$A" --batch 8 --device cuda
.venv/bin/python scripts/04_semantic_topography.py --probe "runs/gs_$A.npz" --n-perm 200
echo "done=$(date -Is)"
