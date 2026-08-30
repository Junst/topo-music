#!/bin/bash
#SBATCH --job-name=topo-rest
#SBATCH --partition=asus_pro6000,gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/rest_%A_%a.out
#SBATCH --error=logs/rest_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# back on a100: torch 2.9.1+cu128 initializes there but fails on the RTX3090 and
# A6000 nodes with "CUDA unknown error", despite driver 580 and /dev/nvidia-uvm
# both being present
case $SLURM_ARRAY_TASK_ID in
  0) .venv/bin/python scripts/14_temporal_accumulation.py --arm matpac_L6 \
       --seconds 20 --batch 4 --repeats 5 --n-perm 1000 --device cuda ;;
  1) .venv/bin/python scripts/14_temporal_accumulation.py --arm pupujepa \
       --seconds 20 --batch 4 --repeats 5 --n-perm 1000 --device cuda ;;
  2) .venv/bin/python scripts/10_transposition_curve.py --arm matpac_L6 \
       --split nsynth-train --max-instruments 150 --anchors-per-group 6 \
       --n-boot 5000 --device cuda --save-emb ;;
esac
echo "done=$(date -Is)"
