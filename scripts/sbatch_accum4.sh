#!/bin/bash
#SBATCH --job-name=topo-acc4
#SBATCH --partition=base_suma_rtx3090,suma_rtx4090,suma_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/acc4_%A_%a.out
#SBATCH --error=logs/acc4_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# gigabyte_a6000 is excluded: four array tasks landed on cs-gpu-01 together and
# all of them failed with "CUDA unknown error ... available devices zero"
ARMS=(matpac_L6 pupujepa)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
.venv/bin/python scripts/14_temporal_accumulation.py --arm "$A" --seconds 20 \
  --batch 4 --repeats 5 --n-perm 1000 --device cuda
echo "done=$(date -Is)"
