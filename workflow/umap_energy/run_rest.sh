#!/bin/bash
cd "$(dirname "$0")"
PY=/share/ijp30/hackathons/aixscience/.venv/bin/python
for b in peptide node_pooled node edge pseudoseq; do
    echo "[$(date +%H:%M:%S)] START $b"
    $PY -u umap_block.py "$b" > "umap_$b.log" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  $b (exit $?)"
done
echo "[$(date +%H:%M:%S)] all complete"; touch run_rest.done
