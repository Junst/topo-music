#!/bin/bash
#SBATCH --job-name=topo-pca
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pca_helix_%j.out
#SBATCH --error=logs/pca_helix_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
echo "host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/28_pca_helix.py
echo "done=$(date -Is)"
