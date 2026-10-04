#!/bin/bash
# All four permutation sweeps, strictly one at a time. Sequential because
# LightGBM already takes 16 of the 32 cores and the perceptron sweeps share one
# GPU, so overlapping them splits the same hardware rather than adding capacity.
# Every step resumes: a combination whose seeds already have run_details.json is
# skipped, so this is safe to stop and restart at any point.
cd "$(dirname "$0")"
PY=/share/ijp30/hackathons/aixscience/.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

for step in "split_C2_mace gbm" "split_C2_mace mlp" "split_B_mace gbm" "split_B_mace mlp"; do
    set -- $step
    log "START $1 $2"
    ( cd "$1" && $PY -u run_perms.py "$2" >> "perms_$2.log" 2>&1 )
    log "DONE  $1 $2 (exit $?)"
done
log "all sweeps complete"
