#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/synthetic_runs/src:${REPO_ROOT}/RAPID/src:${PYTHONPATH:-}"

RECIPES_CONFIG="${SENSITIVITY_RECIPES_CONFIG:-${SCRIPT_DIR}/configs/recipes.example.json}"
GRID_CONFIG="${SENSITIVITY_GRID_CONFIG:-${SCRIPT_DIR}/configs/grid.example.json}"

python -m synthetic_runs.pipelines.sensitivity build-recipes --config "${RECIPES_CONFIG}"
python -m synthetic_runs.pipelines.sensitivity run-grid --config "${GRID_CONFIG}"
