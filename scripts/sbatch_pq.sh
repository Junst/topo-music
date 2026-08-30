#!/bin/bash
#SBATCH --job-name=topo-pq
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pq_%j.out
#SBATCH --error=logs/pq_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
export OMP_NUM_THREADS=16
echo "arm=pq_stft host=$(hostname) start=$(date -Is)"
# pure signal processing, no model: torch.stft on CPU at 44.1 kHz, ~0.43 s/clip
.venv/bin/python scripts/09_clip_key_geometry.py --arm pq_stft --seconds 20 \
  --batch 8 --n-perm 2000 --device cpu
echo "done=$(date -Is)"
