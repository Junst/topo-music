#!/bin/bash
#SBATCH --job-name=topo-wsum-fitmuq
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wsum_fitmuq_%j.out
#SBATCH --error=logs/wsum_fitmuq_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
# MuQ's sweep is L2 L6 L8 L12; tap 0 is the input, so the index matches the L
.venv/bin/python scripts/27_layer_weighted.py --enc muq --stage fit \
  --dataset gs --fixed-layers 2 6 8 12
echo "done=$(date -Is)"
