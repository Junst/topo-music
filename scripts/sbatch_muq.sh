#!/bin/bash
#SBATCH --job-name=topo-muq
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/muq_%A_%a.out
#SBATCH --error=logs/muq_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# MuQ has 12 layers to MERT's 24, so these four sit at the same relative depths
# as the mert_L4 / L12 / L16 / L24 we already report.
ARMS=(muq_L2 muq_L6 muq_L8 muq_L12)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
# transformers 5.16 in .venv breaks MuQ's Wav2Vec2Conformer construction
# (config._attn_implementation); ~/envs/muq pins 4.30 and loads it cleanly.
~/envs/muq/bin/python scripts/09_clip_key_geometry.py --arm "$A" --seconds 20 \
  --batch 8 --n-perm 2000 --device cuda
echo "done=$(date -Is)"
