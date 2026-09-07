#!/bin/bash
#SBATCH --job-name=topo-permnull
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/permnull_%j.out
#SBATCH --error=logs/permnull_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=16
# arms and output come in through --export; a bash array indexed by
# SLURM_ARRAY_TASK_ID did not survive the batch shell here
echo "host=$(hostname) arms=$ARMS out=$OUT start=$(date -Is)"
.venv/bin/python scripts/33_label_permutation_null.py \
  --arms $ARMS --n-perm 200 --out "$OUT"
echo "done=$(date -Is)"
