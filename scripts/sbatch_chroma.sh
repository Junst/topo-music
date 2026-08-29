#!/bin/bash
#SBATCH --job-name=topo-chroma
#SBATCH --partition=base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chroma_%j.out
#SBATCH --error=logs/chroma_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
.venv/bin/python scripts/09_clip_key_geometry.py --arm chroma --seconds 20 \
  --batch 8 --n-perm 2000 --device cpu
