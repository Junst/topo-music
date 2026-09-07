#!/bin/bash
#SBATCH --job-name=topo-wsum-muq
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/wsum_muq_%j.out
#SBATCH --error=logs/wsum_muq_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
echo "host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/27_layer_weighted.py --enc muq --stage extract \
  --dataset gs --device cuda --batch 8
.venv/bin/python scripts/27_layer_weighted.py --enc muq --stage fit --dataset gs
echo "done=$(date -Is)"
