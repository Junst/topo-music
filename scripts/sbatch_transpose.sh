#!/bin/bash
#SBATCH --job-name=topo-tr
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-10
#SBATCH --output=logs/tr_%A_%a.out
#SBATCH --error=logs/tr_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
ARMS=(chroma cqt mert_L0 mert_L4 mert_L8 mert_L12 mert_L16 mert_L20 mert_L24 \
      encodec_32k dac_44k)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
DEV=cuda; [[ "$A" == chroma || "$A" == cqt ]] && DEV=cpu
.venv/bin/python scripts/10_transposition_curve.py --arm "$A" --split nsynth-train \
  --max-instruments 150 --anchors-per-group 6 --n-boot 5000 --device $DEV --save-emb
echo "done=$(date -Is)"
