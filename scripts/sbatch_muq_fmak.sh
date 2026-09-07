#!/bin/bash
#SBATCH --job-name=topo-muqfm
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/muqfm_%j.out
#SBATCH --error=logs/muqfm_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# a100 only. ~/envs/muq pins torch 2.6+cu124, which transformers 5.16 in .venv
# would break MuQ against, but cu124 has no kernels for the Blackwell cards in
# the pro6000 partition, so that is where this arm has to run.
echo "arm=muq_L6 dataset=fmak host=$(hostname) start=$(date -Is)"
~/envs/muq/bin/python scripts/09_clip_key_geometry.py --arm muq_L6 --dataset fmak \
  --seconds 20 --batch 4 --n-perm 2000 --device cuda
echo "done=$(date -Is)"
