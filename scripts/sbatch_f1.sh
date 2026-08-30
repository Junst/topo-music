#!/bin/bash
#SBATCH --job-name=topo-f1
# CPU only: no GPU is used, and the GPU partitions would serialise the array
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/f1_%A_%a.out
#SBATCH --error=logs/f1_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=4
# one arm per task: the probe cost scales with dimension, and MATPAC at 3840
# and HCQT at 2160 dominate a serial pass
ARMS=(cqt cqt_norm cqt_fold hcqt pq_stft chroma encodec_32k \
      mert_L4 mert_L12 mert_L16 mert_L24 \
      muq_L2 muq_L6 muq_L8 muq_L12 \
      matpac_L2 matpac_L6 matpac_L8 matpac_L12 pupujepa)
.venv/bin/python scripts/23_probe_f1.py "${ARMS[$SLURM_ARRAY_TASK_ID]}"
