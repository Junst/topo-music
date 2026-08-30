#!/bin/bash
#SBATCH --job-name=topo-hcqt
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hcqt_%j.out
#SBATCH --error=logs/hcqt_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=16
echo "arm=hcqt host=$(hostname) start=$(date -Is)"
# 6 harmonic CQTs per clip at 60 bins/octave, ~0.76 s/clip on one thread
.venv/bin/python scripts/09_clip_key_geometry.py --arm hcqt --seconds 20 \
  --batch 8 --n-perm 2000 --device cpu
echo "done=$(date -Is)"
