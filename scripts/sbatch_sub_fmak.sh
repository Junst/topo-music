#!/bin/bash
#SBATCH --job-name=topo-subfmak
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/subfmak_%A_%a.out
#SBATCH --error=logs/subfmak_%A_%a.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export OMP_NUM_THREADS=4
# The projection refitted on FMAK keys instead of GiantSteps tonics, then
# applied unchanged to the same NSynth recordings. If octave equivalence still
# appears, the relation cannot be a GiantSteps subgenre correlation, which is
# the confound the Limitations paragraph currently concedes.
ARMS=(cqt chroma encodec_32k mert_L12 matpac_L6 pupujepa)
.venv/bin/python scripts/13_tonal_subspace.py --arm "${ARMS[$SLURM_ARRAY_TASK_ID]}" \
  --key-data fmak --n-random 10
