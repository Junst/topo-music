#!/bin/bash
#SBATCH --job-name=topo-acc2
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/acc2_%A_%a.out
#SBATCH --error=logs/acc2_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# the window sweep for the two arms Fig 1 does not yet cover
ARMS=(muq_L6 encodec_32k)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
PY=.venv/bin/python
[[ "$A" == muq_* ]] && PY=~/envs/muq/bin/python
$PY scripts/14_temporal_accumulation.py --arm "$A" --seconds 20 \
  --batch 8 --repeats 5 --n-perm 1000 --device cuda
echo "done=$(date -Is)"
