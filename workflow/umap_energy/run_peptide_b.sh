#!/bin/bash
cd "$(dirname "$0")"
PY=/share/ijp30/hackathons/aixscience/.venv/bin/python
# wait for the current sequential run to finish (marker file, not a name match)
while [ ! -f run_rest.done ]; do sleep 20; done
echo "[$(date +%H:%M:%S)] START peptide split_B"
$PY -u umap_block.py peptide split_B > umap_peptide_splitB.log 2>&1
echo "[$(date +%H:%M:%S)] DONE (exit $?)"
