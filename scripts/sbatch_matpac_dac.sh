#!/bin/bash
#SBATCH --job-name=topo-mpdac
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/mpdac_%A_%a.out
#SBATCH --error=logs/mpdac_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# MATPAC layers at the same fractions of depth as MERT 4/12/16/24, plus the
# second codec, which so far exists only on the NSynth side
ARMS=(matpac_L2 matpac_L6 matpac_L8 matpac_L12 dac_44k)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/09_clip_key_geometry.py --arm "$A" --seconds 20 \
  --batch 4 --n-perm 2000 --device cuda
echo "done=$(date -Is)"
