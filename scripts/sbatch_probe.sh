#!/bin/bash
#SBATCH --job-name=topo-probe
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/probe_%A_%a.out
#SBATCH --error=logs/probe_%A_%a.err
# A GPU is requested because the QoS rejects GPU-less jobs (QOSMinGRES), not
# because the work needs one: the codec encoders are small convnets over 4096
# four-second notes. It is used anyway rather than left idle.
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODECS=(encodec_32k dac_44k encodec_24k)
C=${CODECS[$SLURM_ARRAY_TASK_ID]}
echo "codec=$C host=$(hostname) start=$(date -Is)"
.venv/bin/python scripts/03_probe_nsynth.py --codec "$C" --batch 32 --device cuda
echo "done=$(date -Is)"
