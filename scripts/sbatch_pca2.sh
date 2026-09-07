#!/bin/bash
#SBATCH --job-name=topo-pca2
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pca2_%j.out
#SBATCH --error=logs/pca2_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
echo "host=$(hostname) start=$(date -Is)"
# the six arms figure 3(b) plots need a PCA bar, on top of the eight already
# measured; the script rewrites the whole JSON, so every arm is passed
.venv/bin/python scripts/28_pca_helix.py --arms \
  mert_L4 mert_L12 mert_L24 muq_L2 muq_L6 muq_L12 \
  matpac_L6 pupujepa encodec_32k cqt pq_stft chroma
echo "done=$(date -Is)"
