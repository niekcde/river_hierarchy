#!/bin/zsh
# Refresh the full RAPID stack after the slope-refit equivalent-resistance pass.
# This runner is deliberately sequential to avoid transient storage exhaustion.

set -euo pipefail

PYTHON="/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python"
OUTPUT_ROOT="/Volumes/PhD/river_hierarchy/network_variants/output"
RAPID_WORK_ROOT="/Volumes/PhD/river_hierarchy/network_variants/rapid_work"
WAIT_PID="${1:-}"
WAIT_EXPERIMENT="${2:-${OUTPUT_ROOT}/37_indep__anchored}"
LOG_DIR="${OUTPUT_ROOT}/overnight_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/rapid_refresh_$(date -u +%Y%m%dT%H%M%SZ).log"

exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

run_step() {
  print -r -- "[$(timestamp)] RUN: $*"
  "$@"
  print -r -- "[$(timestamp)] DONE: $*"
}

if [[ -n "$WAIT_PID" ]]; then
  print -r -- "[$(timestamp)] Waiting for equivalent-resistance PID ${WAIT_PID}."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  print -r -- "[$(timestamp)] Equivalent-resistance PID ${WAIT_PID} exited; validating its audit."
  "$PYTHON" - "$WAIT_EXPERIMENT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = root / "equivalent_resistance" / "state_hierarchy_recompute_summary.json"
payload = json.loads(summary.read_text())
failed = int(payload["n_solver_failed_after"])
if failed:
    raise SystemExit(
        f"Refusing RAPID refresh: {root.name} still has {failed} equivalent-resistance solver failures."
    )
print(f"Validated {root.name}: {payload['n_states_recomputed']} states, zero solver failures.")
PY
fi

typeset -a EXAMPLES
EXAMPLES=(02 03 05 11 13 15 21 22 37 41 45 46 48 54 55 56)

for EXAMPLE in "${EXAMPLES[@]}"; do
  if [[ "$EXAMPLE" == "15" ]]; then
    INDEP_NAME="15_indep_direction130__anchored"
    COMPOUND_NAME="15_compound_bubbles_direction130__anchored"
    ISLAND_NAME="15_island_direction130__anchored"
  else
    INDEP_NAME="${EXAMPLE}_indep__anchored"
    COMPOUND_NAME="${EXAMPLE}_compound_bubbles__anchored"
    ISLAND_NAME="${EXAMPLE}_island__anchored"
  fi

  INDEP_DIR="${OUTPUT_ROOT}/${INDEP_NAME}"
  COMPOUND_DIR="${OUTPUT_ROOT}/${COMPOUND_NAME}"
  ISLAND_DIR="${OUTPUT_ROOT}/${ISLAND_NAME}"

  print -r -- "[$(timestamp)] ===== Example ${EXAMPLE} ====="

  run_step "$PYTHON" -B RAPID/run_refresh_experiment_from_manifest.py "$INDEP_DIR" \
    --run --compact-qout --write-mapping-parquet --drop-prepared-inputs
  run_step "$PYTHON" -B RAPID/run_refresh_experiment_from_manifest.py "$COMPOUND_DIR" \
    --run --compact-qout --write-mapping-parquet --drop-prepared-inputs

  ANALYSIS_MANIFEST="${ISLAND_DIR}/island_analysis_manifest_strict.csv"
  BASE_LINK_CONTEXT="${INDEP_DIR}/states/S000_base/hydraulic_context/link_hydraulic_context.csv"
  BASE_NODE_CONTEXT="${INDEP_DIR}/states/S000_base/hydraulic_context/node_hydraulic_context.csv"
  REFERENCE_SECTION_SUMMARY="${INDEP_DIR}/reference_section/reference_section_summary.json"
  ISLAND_CONTEXT_REGISTRY="${ISLAND_DIR}/island_hydraulic_context_registry.csv"
  ISLAND_PREP_REGISTRY="${ISLAND_DIR}/island_rapid_prep_registry.csv"

  run_step "$PYTHON" -B network_variants/run_generate_island_hydraulic_contexts.py \
    "$ANALYSIS_MANIFEST" \
    --base-link-context "$BASE_LINK_CONTEXT" \
    --base-node-context "$BASE_NODE_CONTEXT" \
    --refresh-existing \
    --slope-profile-mode reference_section \
    --reference-section-summary-path "$REFERENCE_SECTION_SUMMARY" \
    --max-slope-for-k 0.009 \
    --section-slope-ratio-max 5
  run_step "$PYTHON" -B network_variants/run_island_equivalent_resistance.py \
    "$ANALYSIS_MANIFEST" "$INDEP_DIR"
  run_step "$PYTHON" -B RAPID/run_prepare_island_observations.py \
    "$ANALYSIS_MANIFEST" "$ISLAND_CONTEXT_REGISTRY" "$INDEP_DIR" \
    --work-root "${RAPID_WORK_ROOT}/${ISLAND_NAME}"
  run_step "$PYTHON" -B RAPID/run_island_observations.py "$ISLAND_PREP_REGISTRY" \
    --compact-qout --write-mapping-parquet --drop-prepared-inputs
  run_step "$PYTHON" -B RAPID/compare_island_observations.py \
    "$ANALYSIS_MANIFEST" "$INDEP_DIR"
done

print -r -- "[$(timestamp)] All overnight RAPID refresh steps completed successfully."
