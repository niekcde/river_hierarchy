#!/bin/bash
set -euo pipefail
cd /Users/6256481/Code/river-hierarchy
PYTHON=/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python
OUTPUT=/Volumes/PhD/river_hierarchy/three_run_comparison_20261028
"$PYTHON" -c 'import os; assert os.path.ismount("/Volumes/PhD"), "PhD drive is not mounted"'
# Check inputs and free space before creating or appending the batch log.
"$PYTHON" RAPID/investigations/cap_investigation_20261020/run_three_runs_control_fix.py --output "$OUTPUT" "$@"
mkdir -p "$OUTPUT"
caffeinate -i env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$PYTHON" -u RAPID/investigations/cap_investigation_20261020/run_three_runs_control_fix.py \
  --output "$OUTPUT" --run "$@" 2>&1 | tee -a "$OUTPUT/run.log"
