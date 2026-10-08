"""Run and audit numerical continuation for one stored bubble resistance domain.

This is deliberately an investigation runner. It writes no hierarchy or RAPID
products and cannot promote a continuation result into an experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchy_level_definition.metrics.equivalent_resistance import (
    solve_bubble_equivalent_K_continuation,
)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _neutral_heads(node_ids: pd.Series, source: int, sink: int) -> dict[int, float]:
    heads = {int(node): 0.5 for node in pd.to_numeric(node_ids, errors="coerce").dropna().astype(int)}
    heads[source], heads[sink] = 1.0, 0.0
    return heads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--state-id", default="S000_base")
    parser.add_argument("--bubble-id", required=True, type=int)
    parser.add_argument("--eps", nargs="+", type=float, default=(1e-4, 1e-6, 1e-8, 1e-10))
    parser.add_argument("--energy-max-iterations", type=int, default=20_000)
    parser.add_argument("--energy-max-function-evaluations", type=int, default=50_000)
    parser.add_argument("--polish-max-function-evaluations", type=int, default=4_000)
    parser.add_argument(
        "--initialization",
        choices=("available", "neutral", "both"),
        default="both",
        help="Which independent initialization(s) to audit; default: both.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.experiment_dir.expanduser().resolve()
    state = root / "states" / args.state_id
    hierarchy = state / "hierarchy"
    metrics = pd.read_csv(hierarchy / "bubble_equivalent_metrics.csv")
    metric = metrics.loc[pd.to_numeric(metrics["compound_bubble_id"], errors="coerce").eq(args.bubble_id)]
    if len(metric) != 1:
        raise ValueError(f"Expected exactly one bubble metric row, found {len(metric)}.")
    metric = metric.iloc[0]
    edges = pd.read_csv(hierarchy / "bubble_edge_resistance.csv")
    edges = edges.loc[pd.to_numeric(edges["compound_bubble_id"], errors="coerce").eq(args.bubble_id)].copy()
    node_context = pd.read_csv(state / "hydraulic_context" / "node_hydraulic_context.csv")
    incident = set(pd.concat([edges["u"], edges["v"]], ignore_index=True).astype(int))
    nodes = node_context.loc[pd.to_numeric(node_context["id_node"], errors="coerce").isin(incident)].copy()
    nodes = nodes.rename(columns={"id_node": "node_id"})
    source, sink = int(metric["upstream_boundary_node"]), int(metric["downstream_boundary_node"])
    output = args.output_dir or (
        root / "equivalent_resistance" / "continuation_investigation" /
        f"{args.state_id}_bubble_{args.bubble_id:03d}"
    )
    output.mkdir(parents=True, exist_ok=True)

    available_head_columns = ("H", "head", "sword_wse", "wse", "h")
    available_guess = (
        nodes
        if any(column in nodes.columns for column in available_head_columns)
        else None
    )
    all_attempts = {
        "available_node_heads": available_guess,
        "neutral_internal_heads": _neutral_heads(nodes["node_id"], source, sink),
    }
    requested_labels = {
        "available": ("available_node_heads",),
        "neutral": ("neutral_internal_heads",),
        "both": tuple(all_attempts),
    }[args.initialization]
    attempts = {}
    for label in requested_labels:
        guess = all_attempts[label]
        result = solve_bubble_equivalent_K_continuation(
            nodes,
            edges,
            upstream_node=source,
            downstream_node=sink,
            optional_H_guess=guess,
            eps_schedule=args.eps,
            solver_backend="energy_sparse",
            energy_max_iterations=args.energy_max_iterations,
            energy_max_function_evaluations=args.energy_max_function_evaluations,
            polish_max_function_evaluations=args.polish_max_function_evaluations,
        )
        attempts[label] = {
            key: value for key, value in result.items()
            if key not in {"node_heads", "edge_flows", "internal_heads"}
        }
        result["node_heads"].to_csv(output / f"{label}_node_heads.csv", index=False)
        result["edge_flows"].to_csv(output / f"{label}_edge_flows.csv", index=False)

    successful = [value for value in attempts.values() if value["solver_success"]]
    summary = {
        "schema_version": "equivalent-resistance-continuation-investigation-v1",
        "experiment_dir": str(root),
        "state_id": args.state_id,
        "compound_bubble_id": args.bubble_id,
        "source_node": source,
        "sink_node": sink,
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "eps_schedule": list(args.eps),
        "initialization_mode": args.initialization,
        "n_successful_attempts": len(successful),
        "attempts": attempts,
    }
    _write_json(output / "continuation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
