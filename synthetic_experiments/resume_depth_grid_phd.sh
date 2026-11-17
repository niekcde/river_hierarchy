#!/bin/bash
set -euo pipefail
cd /Users/6256481/Code/river-hierarchy
PYTHON=/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python
OUTPUT=/Volumes/SyntheticResults20261027
IMAGE=/Volumes/PhD/river_hierarchy/synthetic_experiments/synthetic_results_20261027.sparsebundle
if ! "$PYTHON" -c 'import os,sys; sys.exit(0 if os.path.ismount(sys.argv[1]) else 1)' "$OUTPUT"; then
  /usr/bin/hdiutil attach -nobrowse "$IMAGE"
fi
"$PYTHON" -c 'import os,sys; assert os.path.ismount(sys.argv[1]), "Expected results volume is not mounted"' "$OUTPUT"
[ -f "$OUTPUT/manifest.json" ] || { echo 'Existing manifest missing; refusing a new run'; exit 1; }
caffeinate -i env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$PYTHON" synthetic_experiments/run_depth_comparison.py \
  --config synthetic_experiments/configs/depth_comparison_20261027.json \
  --output "$OUTPUT" --factorial --workers 2 --run \
  2>&1 | tee -a "$OUTPUT/run.log"
