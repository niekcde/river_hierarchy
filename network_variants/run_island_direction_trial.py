"""Run one isolated parent-direction sensitivity trial for an island state.

This deliberately does *not* modify a reviewed parent network or an existing
island experiment.  It materialises a corrected parent-link copy, reruns only
the selected island's mask/RivGraph/direction stages, and records enough
provenance to decide whether a parent correction should be promoted later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchy_level_definition.manual_review.apply_link_reversals import apply_reversals
from network_variants.island_experiment import (
    direct_island_topology,
    load_island_variant_selection,
    run_island_topology,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directed_topology_summary(links_path: Path) -> dict[str, object]:
    links = gpd.read_file(links_path)
    graph = nx.DiGraph()
    for row in links[["id_link", "id_us_node", "id_ds_node"]].itertuples(index=False):
        graph.add_edge(int(row.id_us_node), int(row.id_ds_node), id_link=int(row.id_link))
    sources = sorted(int(node) for node in graph.nodes if graph.in_degree(node) == 0)
    sinks = sorted(int(node) for node in graph.nodes if graph.out_degree(node) == 0)
    cycles = list(nx.simple_cycles(graph)) if not nx.is_directed_acyclic_graph(graph) else []
    return {
        "n_links": int(len(links)),
        "is_dag": bool(not cycles),
        "n_sources": len(sources),
        "n_sinks": len(sinks),
        "source_node_ids": sources,
        "sink_node_ids": sinks,
        "cycle_node_sequences": cycles[:10],
        "is_single_inlet_single_outlet_dag": bool(not cycles and len(sources) == 1 and len(sinks) == 1),
    }


def finalize_existing_direction_trial(output_dir: str | Path) -> dict[str, object]:
    """Recover a completed trial whose final audit write was interrupted.

    This intentionally accepts only a complete retained directed result and
    cannot regenerate, overwrite, or promote a parent network.
    """
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "trial_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Trial manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    island_id = str(payload.get("island_id", ""))
    if not island_id:
        raise ValueError(f"Trial manifest has no island_id: {manifest_path}")
    directed_dir = root / "island_rerun" / island_id / "variant" / "directed"
    links_path = directed_dir / "directed_links.gpkg"
    nodes_path = directed_dir / "directed_nodes.gpkg"
    summary_path = directed_dir / "anchor_summary.json"
    if not all(path.is_file() for path in (links_path, nodes_path, summary_path)):
        raise FileNotFoundError(
            "Trial cannot be finalized because its retained directed products are incomplete: "
            f"{directed_dir}"
        )
    topology_summary = _directed_topology_summary(links_path)
    parent_links_path = Path(str(payload["trial_parent_links"])).expanduser().resolve()
    if not parent_links_path.is_file():
        raise FileNotFoundError(f"Trial corrected parent links are missing: {parent_links_path}")
    payload.update({
        "status": "complete" if topology_summary["is_single_inlet_single_outlet_dag"] else "invalid_topology",
        "trial_parent_topology_summary": _directed_topology_summary(parent_links_path),
        "topology_summary": topology_summary,
        "directed_links_path": str(links_path),
        "directed_nodes_path": str(nodes_path),
        "direction_summary_path": str(summary_path),
    })
    payload.pop("error", None)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return payload


def run_island_direction_trial(
    *,
    island_observation_manifest: str | Path,
    island_id: str,
    parent_links_path: str | Path,
    parent_nodes_path: str | Path,
    reversed_parent_link_ids: list[int],
    base_mask_path: str | Path,
    output_dir: str | Path,
    exit_sides: str,
    example_id: str,
    sword_node_source_path: str | Path | None = None,
    sword_wse_field: str | None = None,
    sword_match_tolerance: float | None = None,
    sword_example_station_source_path: str | Path | None = None,
    sword_station_match_source_path: str | Path | None = None,
    sword_reach_buffer_steps: int = 2,
) -> dict[str, object]:
    """Create one non-destructive parent-direction trial and validate its output."""
    manifest_path = Path(island_observation_manifest).expanduser().resolve()
    parent_links = Path(parent_links_path).expanduser().resolve()
    parent_nodes = Path(parent_nodes_path).expanduser().resolve()
    base_mask = Path(base_mask_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    for label, path in {
        "island observation manifest": manifest_path,
        "parent links": parent_links,
        "parent nodes": parent_nodes,
        "base mask": base_mask,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Trial output must be new or empty: {root}")
    if not reversed_parent_link_ids:
        raise ValueError("At least one --reverse-parent-link-id is required.")

    observations = pd.read_csv(manifest_path, low_memory=False)
    target = observations.loc[observations["island_id"].astype(str).eq(str(island_id))]
    if len(target) != 1:
        raise ValueError(f"Expected exactly one {island_id!r} row in {manifest_path}, found {len(target)}.")
    row = target.iloc[0]
    if str(row.get("execution_action")) != "run_island_state":
        raise ValueError(f"Island {island_id} is not a genuine island state: {row.get('execution_action')!r}")

    root.mkdir(parents=True, exist_ok=True)
    parent_trial_dir = root / "parent_direction_trial"
    parent_trial_dir.mkdir()
    trial_parent_links = parent_trial_dir / parent_links.name
    source_links = gpd.read_file(parent_links)
    corrected_links, reversal_details = apply_reversals(
        source_links,
        {int(link_id) for link_id in reversed_parent_link_ids},
    )
    corrected_links.to_file(trial_parent_links, driver="GPKG")
    reversal_details.to_csv(parent_trial_dir / "parent_link_reversals.csv", index=False)

    trial_state = root / "island_rerun" / str(island_id)
    result: dict[str, object] = {
        "schema_version": "island-parent-direction-trial-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "example_id": str(example_id),
        "island_id": str(island_id),
        "source_observation_manifest": str(manifest_path),
        "source_parent_links": str(parent_links),
        "source_parent_links_sha256": _sha256(parent_links),
        "trial_parent_links": str(trial_parent_links),
        "trial_parent_links_sha256": _sha256(trial_parent_links),
        "trial_parent_topology_summary": _directed_topology_summary(trial_parent_links),
        "parent_nodes": str(parent_nodes),
        "base_mask": str(base_mask),
        "reversed_parent_link_ids": sorted({int(link_id) for link_id in reversed_parent_link_ids}),
        "reversal_details": reversal_details.to_dict(orient="records"),
        "trial_state_dir": str(trial_state),
    }
    try:
        selection = load_island_variant_selection(row["island_setup_dir"], str(island_id))
        with rasterio.open(base_mask) as source:
            match_tolerance = abs(float(source.transform.a)) * 1.25
        topology = run_island_topology(
            selection,
            base_mask_path=base_mask,
            output_dir=trial_state / "variant",
            exit_sides=exit_sides,
        )
        directed = direct_island_topology(
            topology,
            parent_links_path=trial_parent_links,
            parent_nodes_path=parent_nodes,
            output_dir=trial_state / "variant" / "directed",
            match_tolerance=match_tolerance,
            sword_node_source_path=sword_node_source_path,
            sword_wse_field=sword_wse_field,
            sword_match_tolerance=sword_match_tolerance,
            example_id=str(example_id),
            sword_example_station_source_path=sword_example_station_source_path,
            sword_station_match_source_path=sword_station_match_source_path,
            sword_reach_buffer_steps=sword_reach_buffer_steps,
        )
        topology_summary = _directed_topology_summary(directed.directed_links_path)
        result.update({
            "status": "complete" if topology_summary["is_single_inlet_single_outlet_dag"] else "invalid_topology",
            "topology_summary": topology_summary,
            "directed_links_path": str(directed.directed_links_path),
            "directed_nodes_path": str(directed.directed_nodes_path),
            "direction_summary_path": str(directed.anchor_summary_path),
        })
    except Exception as exc:
        result.update({"status": "failed", "error": str(exc)})

    output_manifest = root / "trial_manifest.json"
    output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a non-destructive parent-direction sensitivity trial for one genuine island state."
    )
    parser.add_argument("--island-observation-manifest", required=True)
    parser.add_argument("--island-id", required=True)
    parser.add_argument("--parent-links", required=True)
    parser.add_argument("--parent-nodes", required=True)
    parser.add_argument("--reverse-parent-link-id", type=int, action="append", required=True)
    parser.add_argument("--base-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exit-sides", required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--sword-node-source")
    parser.add_argument("--sword-wse-field")
    parser.add_argument("--sword-match-tolerance", type=float)
    parser.add_argument("--sword-example-stations-source")
    parser.add_argument("--sword-station-match-source")
    parser.add_argument("--sword-reach-buffer-steps", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_island_direction_trial(
        island_observation_manifest=args.island_observation_manifest,
        island_id=args.island_id,
        parent_links_path=args.parent_links,
        parent_nodes_path=args.parent_nodes,
        reversed_parent_link_ids=args.reverse_parent_link_id,
        base_mask_path=args.base_mask,
        output_dir=args.output_dir,
        exit_sides=args.exit_sides,
        example_id=args.example_id,
        sword_node_source_path=args.sword_node_source,
        sword_wse_field=args.sword_wse_field,
        sword_match_tolerance=args.sword_match_tolerance,
        sword_example_station_source_path=args.sword_example_stations_source,
        sword_station_match_source_path=args.sword_station_match_source,
        sword_reach_buffer_steps=args.sword_reach_buffer_steps,
    )
    print(f"Trial status: {result['status']}")
    print(f"Trial manifest: {Path(args.output_dir).expanduser().resolve() / 'trial_manifest.json'}")
    if result["status"] != "complete":
        print(result.get("error", result.get("topology_summary", "Unknown trial failure")), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
