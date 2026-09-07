#!/bin/bash
#SBATCH --job-name=topo-helix
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/helicality_%j.out
#SBATCH --error=logs/helicality_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
echo "host=$(hostname) start=$(date -Is)"
echo "--- fitter selftest ---"
.venv/bin/python scripts/29_helicality.py --selftest
echo "--- arms ---"
.venv/bin/python scripts/29_helicality.py
echo "done=$(date -Is)"
