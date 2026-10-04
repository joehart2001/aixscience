#!/bin/bash
# node and edge after the boltz run finishes. Sequential: UMAP is single
# threaded when seeded, and the node/edge blocks are 11,520 and 14,450 wide,
# so running them together would only contend for memory bandwidth.
cd "$(dirname "$0")"
PY=/share/ijp30/hackathons/aixscience/.venv/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

while pgrep -f "[u]map_block.py" > /dev/null; do sleep 20; done
log "boltz finished"
for b in node edge; do
    log "START $b"
    $PY -u umap_block.py "$b" > "umap_$b.log" 2>&1
    log "DONE  $b (exit $?)"
done
log "all UMAPs complete"
