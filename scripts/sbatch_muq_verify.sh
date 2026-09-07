#!/bin/bash
#SBATCH --job-name=topo-muq-verify
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/muq_verify_%j.out
#SBATCH --error=logs/muq_verify_%j.err
set -euo pipefail
cd /scratch2/solbon1212/topo-music
export HF_HOME=/scratch2/solbon1212/hf_cache
echo "host=$(hostname) start=$(date -Is)"

# 1. script 27 through the shared helper and MuQ's own .hidden_states, the
#    exact access pattern scripts 09, 10 and 14 use
.venv/bin/python scripts/27_layer_weighted.py --enc muq --stage extract \
  --dataset gs --device cuda --batch 8

# 2. the tensors must match the hook-only version bit for bit
.venv/bin/python - <<'PY'
import numpy as np
a = np.load("runs/_layers_muq_hookver.npz")["Z"]
b = np.load("runs/layers_muq.npz")["Z"]
assert a.shape == b.shape, (a.shape, b.shape)
print("max abs diff", float(np.abs(a.astype(np.float32)
                                   - b.astype(np.float32)).max()))
assert np.array_equal(a, b), "helper changed the embeddings"
print("IDENTICAL to the hook-only extraction")
PY

# 3. script 09 itself, on one split and a token permutation count, written to a
#    scratch path so the cached arm is untouched
.venv/bin/python scripts/09_clip_key_geometry.py --arm muq_L6 \
  --splits GS.test.jsonl --n-perm 5 --device cuda \
  --out runs/_smoke_muq09.json

# 4. and its embeddings must match the cached full run on the same clips
.venv/bin/python - <<'PY'
import numpy as np
new = np.load("runs/_smoke_muq09.npz")
old = np.load("runs/clipkey_muq_L6.npz")
Zn, Zo = new["Z"], old["Z"][-len(new["Z"]):]
print("smoke rows", Zn.shape, "compared against last", Zo.shape, "cached rows")
print("max abs diff", float(np.abs(Zn - Zo).max()))
assert np.allclose(Zn, Zo, atol=1e-4), "script 09 muq path drifted"
print("script 09 muq path reproduces the cached embeddings")
PY
echo "done=$(date -Is)"
