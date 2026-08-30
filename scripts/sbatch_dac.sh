#!/bin/bash
#SBATCH --job-name=topo-dac
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dac_%j.out
#SBATCH --error=logs/dac_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
echo "arm=dac_44k host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/09_clip_key_geometry.py --arm dac_44k --seconds 20 \
  --batch 8 --n-perm 2000 --device cuda
echo "done=$(date -Is)"
