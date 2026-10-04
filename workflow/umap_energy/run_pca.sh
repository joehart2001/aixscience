#!/bin/bash
cd "$(dirname "$0")"
PY=/share/ijp30/hackathons/aixscience/.venv/bin/python
for b in energy boltz; do
  echo "[$(date +%H:%M:%S)] START $b (forced PCA 50)"
  FORCE_PCA=1 $PY -u umap_block.py "$b" > "umap_${b}_pca50.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $b (exit $?)"
done
