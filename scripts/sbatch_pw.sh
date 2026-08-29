#!/bin/bash
#SBATCH --job-name=topo-pw
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/pw_%A_%a.out
#SBATCH --error=logs/pw_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
ARMS=(chroma cqt encodec_32k mert_L4 mert_L12 mert_L16 mert_L24)
.venv/bin/python scripts/11_probe_weight_from_cache.py "${ARMS[$SLURM_ARRAY_TASK_ID]}"
