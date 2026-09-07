#!/bin/bash
#SBATCH --job-name=topo-pca-hcqt
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pca_hcqt_%j.out
#SBATCH --error=logs/pca_hcqt_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
.venv/bin/python scripts/28_pca_helix.py --arms \
  mert_L4 mert_L12 mert_L24 muq_L2 muq_L6 muq_L12 \
  matpac_L6 pupujepa encodec_32k cqt pq_stft chroma hcqt
