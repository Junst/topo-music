#!/bin/bash
#SBATCH --job-name=topo-wsum-fit
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wsum_fit_%j.out
#SBATCH --error=logs/wsum_fit_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
# MERT's four-point sweep is L4 L12 L16 L24; MATPAC's is L2 L6 L8 L12. The
# tap index matches the L number for MERT (tap 0 is the feature projection)
# and is one less for MATPAC, whose taps start at layer 1.
.venv/bin/python scripts/27_layer_weighted.py --enc mert --stage fit \
  --dataset gs --fixed-layers 4 12 16 24
.venv/bin/python scripts/27_layer_weighted.py --enc matpac --stage fit \
  --dataset gs --fixed-layers 1 5 7 11
echo "done=$(date -Is)"
