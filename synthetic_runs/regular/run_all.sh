#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/synthetic_runs/src:${REPO_ROOT}/RAPID/src:${PYTHONPATH:-}"

GEOM_CONFIG="${REGULAR_GEOMETRY_CONFIG:-${SCRIPT_DIR}/configs/geometry.example.json}"
SAMPLING_CONFIG="${REGULAR_SAMPLING_CONFIG:-${SCRIPT_DIR}/configs/sampling.example.json}"
ROUTING_CONFIG="${REGULAR_ROUTING_CONFIG:-${SCRIPT_DIR}/configs/routing.example.json}"
K_METRICS_CONFIG="${REGULAR_K_METRICS_CONFIG:-${SCRIPT_DIR}/configs/k_metrics.example.json}"

python -m synthetic_runs.pipelines.regular build-geometry --config "${GEOM_CONFIG}"
python -m synthetic_runs.pipelines.regular sample-widths --config "${SAMPLING_CONFIG}"
python -m synthetic_runs.pipelines.regular route --config "${ROUTING_CONFIG}"
python -m synthetic_runs.pipelines.regular k-metrics --config "${K_METRICS_CONFIG}"
