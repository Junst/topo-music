#!/bin/bash
#SBATCH --job-name=topo-acc3
# a100 has three nodes and queues; these fit in 24 GB and the RTX/A6000
# partitions have an order of magnitude more of them
#SBATCH --partition=base_suma_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000
#SBATCH --qos=base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/acc3_%A_%a.out
#SBATCH --error=logs/acc3_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# the window sweep for the arms Fig 1 does not yet cover
ARMS=(matpac_L6 pupujepa pq_stft hcqt)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
DEV=cuda; [[ "$A" == pq_stft || "$A" == hcqt ]] && DEV=cpu
.venv/bin/python scripts/14_temporal_accumulation.py --arm "$A" --seconds 20 \
  --batch 4 --repeats 5 --n-perm 1000 --device $DEV
echo "done=$(date -Is)"
