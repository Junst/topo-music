#!/bin/bash
#SBATCH --job-name=topo-f1fm
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/f1fm_%A_%a.out
#SBATCH --error=logs/f1fm_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=4
# one arm per task: probe cost scales with dimension and MATPAC at 3840 would
# dominate a serial pass, as it did on GiantSteps
ARMS=(chroma cqt hcqt pq_stft encodec_32k mert_L12 muq_L6 matpac_L6 pupujepa)
.venv/bin/python scripts/23_probe_f1.py --fmak "${ARMS[$SLURM_ARRAY_TASK_ID]}"
