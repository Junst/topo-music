#!/bin/bash
#SBATCH --job-name=topo-muqtr
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/muqtr_%A_%a.out
#SBATCH --error=logs/muqtr_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
ARMS=(muq_L2 muq_L6 muq_L8 muq_L12)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
# same settings as the MERT arms in sbatch_transpose.sh, so the two are
# comparable note for note; --save-emb is needed by the subspace transfer
~/envs/muq/bin/python scripts/10_transposition_curve.py --arm "$A" --split nsynth-train \
  --max-instruments 150 --anchors-per-group 6 --n-boot 5000 --device cuda --save-emb
echo "done=$(date -Is)"
