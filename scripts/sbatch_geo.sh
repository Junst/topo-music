#!/bin/bash
#SBATCH --job-name=topo-geo
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/geo_%A_%a.out
#SBATCH --error=logs/geo_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=4
ARMS=(cqt chroma encodec_32k mert_L4 mert_L12 mert_L24 muq_L2 muq_L6 muq_L12)
.venv/bin/python scripts/25_geodesic_octave.py --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
