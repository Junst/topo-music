#!/bin/bash
#SBATCH --job-name=topo-geo2
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/geo2_%A_%a.out
#SBATCH --error=logs/geo2_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=4
# the two 60-bin-per-octave spectral arms, which the first geodesic sweep
# predates; PQ-STFT is the only arm with substantial native octave structure,
# so its geodesic value is the one missing from the metric comparison
ARMS=(pq_stft hcqt)
.venv/bin/python scripts/25_geodesic_octave.py --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}"
