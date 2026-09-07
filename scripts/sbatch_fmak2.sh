#!/bin/bash
#SBATCH --job-name=topo-fmak2
#SBATCH --partition=asus_pro6000,gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/fmak2_%A_%a.out
#SBATCH --error=logs/fmak2_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# the layers Fig 2 uses, so the FMAK-fitted projection can be compared with the
# GiantSteps-fitted one on the same arms
ARMS=(mert_L4 mert_L24 muq_L2 muq_L12)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
PY=.venv/bin/python; [[ "$A" == muq_* ]] && PY=~/envs/muq/bin/python
$PY scripts/09_clip_key_geometry.py --arm "$A" --dataset fmak \
  --seconds 20 --batch 4 --n-perm 2000 --device cuda
echo "done=$(date -Is)"
