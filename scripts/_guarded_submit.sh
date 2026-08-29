#!/bin/bash
# The controller intermittently answers sbatch with "Unexpected message received"
# on submissions it has in fact accepted; a naive retry loop once cost ~1.6 GPU
# hours in duplicate arrays. So: wait until squeue answers cleanly, then submit
# a job name only if no job by that name is already queued or running.
set -u
cd /scratch2/solbon1212/topo-music
DEADLINE=$(( $(date +%s) + 21600 ))
declare -A WANT=( [topo-fold]=scripts/sbatch_fold.sh [topo-tr]=scripts/sbatch_transpose.sh [topo-pw]=scripts/sbatch_pw.sh )
while [ $(date +%s) -lt $DEADLINE ]; do
  if Q=$(squeue -u "$USER" -h -o "%j" 2>/dev/null); then
    for name in "${!WANT[@]}"; do
      [ -z "${WANT[$name]:-}" ] && continue
      if grep -qx "$name" <<<"$Q"; then
        echo "$(date -Is) $name already present, not submitting"; WANT[$name]=""
      else
        echo "$(date -Is) submitting $name"
        sbatch "${WANT[$name]}" && WANT[$name]=""
      fi
    done
    [ -z "${WANT[topo-fold]}${WANT[topo-tr]}${WANT[topo-pw]}" ] && { echo "all handled"; exit 0; }
  fi
  sleep 60
done
echo "deadline reached with pending: ${!WANT[*]}"
