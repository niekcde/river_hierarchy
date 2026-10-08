"""Write a non-destructive exact-reduction audit for a stored bubble domain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.equivalent_resistance_reduction import (
    reduce_series_parallel_resistance_graph,
)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--state-id", default="S000_base")
    parser.add_argument("--bubble-id", required=True, type=int)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.experiment_dir.expanduser().resolve()
    state_dir = root / "states" / args.state_id
    source = state_dir / "hierarchy" / "bubble_edge_resistance.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Bubble resistance edges are missing: {source}")
    metrics_path = state_dir / "hierarchy" / "bubble_equivalent_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metric = metrics.loc[
        pd.to_numeric(metrics["compound_bubble_id"], errors="coerce").eq(args.bubble_id)
    ]
    if len(metric) != 1:
        raise ValueError(f"Expected one metric row for bubble {args.bubble_id}, found {len(metric)}.")
    metric = metric.iloc[0]
    edges = pd.read_csv(source)
    domain = edges.loc[
        pd.to_numeric(edges["compound_bubble_id"], errors="coerce").eq(args.bubble_id)
    ].copy()
    if domain.empty:
        raise ValueError(f"No resistance edges found for bubble {args.bubble_id}.")
    result = reduce_series_parallel_resistance_graph(
        domain,
        source_node=int(metric["upstream_boundary_node"]),
        sink_node=int(metric["downstream_boundary_node"]),
    )
    output = args.output_dir or (
        root / "equivalent_resistance" / "reduction_audit" /
        f"{args.state_id}_bubble_{args.bubble_id:03d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    result.reduced_edges.to_csv(output / "reduced_edges.csv", index=False)
    result.operations.to_csv(output / "reduction_operations.csv", index=False)
    summary = {
        **result.summary,
        "experiment_dir": str(root),
        "state_id": args.state_id,
        "compound_bubble_id": args.bubble_id,
        "source_edge_file": str(source),
        "source_metric_file": str(metrics_path),
        "output_dir": str(output),
    }
    _write_json(output / "reduction_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
