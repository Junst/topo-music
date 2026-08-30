#!/bin/bash
#SBATCH --job-name=topo-tr2
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/tr2_%A_%a.out
#SBATCH --error=logs/tr2_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# NSynth is missing for the two encoders added last, which leaves them in
# Table 1 but out of the octave analysis
ARMS=(matpac_L6)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/10_transposition_curve.py --arm "$A" --split nsynth-train \
  --max-instruments 150 --anchors-per-group 6 --n-boot 5000 --device cuda --save-emb
echo "done=$(date -Is)"
