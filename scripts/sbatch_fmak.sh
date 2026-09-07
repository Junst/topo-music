#!/bin/bash
#SBATCH --job-name=topo-fmak
#SBATCH --partition=asus_pro6000,gigabyte_pro6000
#SBATCH --qos=pro6000_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/fmak_%A_%a.out
#SBATCH --error=logs/fmak_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
# the same arms Table 1 prints, on FMAK: 5489 expert-labeled tracks over 17
# genres, one excerpt per track, so key is not confounded with EDM subgenre and
# no track spans the split
ARMS=(chroma cqt hcqt pq_stft encodec_32k mert_L12 muq_L6 matpac_L6 pupujepa)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
echo "arm=$A host=$(hostname) start=$(date -Is)"
DEV=cuda; [[ "$A" == chroma || "$A" == cqt || "$A" == hcqt || "$A" == pq_stft ]] && DEV=cpu
# MuQ builds a Wav2Vec2ConformerEncoder that transformers 5.16 in .venv rejects;
# ~/envs/muq pins 4.30 and loads it, and it initializes CUDA on pro6000
PY=.venv/bin/python; [[ "$A" == muq_* ]] && PY=~/envs/muq/bin/python
$PY scripts/09_clip_key_geometry.py --arm "$A" --dataset fmak \
  --seconds 20 --batch 4 --n-perm 2000 --device $DEV
echo "done=$(date -Is)"
