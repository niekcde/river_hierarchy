"""Auditable physical-length retrofit for retained network-variant hierarchies.

This intentionally rebuilds only hierarchy tables from retained final graph and
width products.  It never invokes mask generation, RivGraph, width estimation,
hydraulic context, RAPID, or equivalent-resistance solvers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from hierarchy_level_definition.run_unit_workflow import run_unit_workflow


RETROFIT_VERSION = "physical-centreline-length-v1"
POLICY = {
    "preferred_length_field": "len",
    "legacy_diagnostic_field": "len_adj",
    "excluded_workflows": [
        "masks", "RivGraph", "width calculation", "hydraulic-context",
        "RAPID", "equivalent-resistance",
    ],
}
HIERARCHY_FILES = (
    "unit_summary.csv", "path_metrics.csv", "unit_metrics.csv",
    "hierarchy_level_metrics.csv", "collapse_ranking.csv", "merge_tree.csv",
    "ordered_group_partitions.csv", "group_count_summary.csv", "bubble_summary.csv",
    "selected_groups.csv",
)
KEY_COLUMNS = {
    "unit_summary.csv": ("unit_id",), "path_metrics.csv": ("unit_id", "path_id"),
    "unit_metrics.csv": ("unit_id",), "collapse_ranking.csv": ("unit_id",),
    "hierarchy_level_metrics.csv": ("collapse_level",), "bubble_summary.csv": ("compound_bubble_id",),
    "merge_tree.csv": ("n_groups", "group_index"),
    "ordered_group_partitions.csv": ("n_groups", "group_index"),
    "group_count_summary.csv": ("n_groups",), "selected_groups.csv": ("n_groups", "group_index"),
}


@dataclass(frozen=True)
class RetrofitTarget:
    hierarchy_dir: Path
    links_path: Path | None
    nodes_path: Path | None
    manifest_path: Path | None
    safe_to_recompute: bool
    reason: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if not path.name.startswith(".") and path.is_file()), None)


def discover_targets(output_root: str | Path) -> list[RetrofitTarget]:
    """Find every retained hierarchy output and its final reusable inputs."""
    root = Path(output_root).expanduser().resolve()
    targets: list[RetrofitTarget] = []
    # Output trees contain large solve caches and macOS sidecar files.  Traverse
    # only the retained state topology rather than recursively walking caches.
    hierarchy_dirs: list[Path] = []
    experiment_dirs = [root] if (root / "states").is_dir() else list(root.iterdir())
    for experiment_dir in experiment_dirs:
        states_dir = experiment_dir / "states"
        if not experiment_dir.is_dir() or experiment_dir.name.startswith(".") or not states_dir.is_dir():
            continue
        hierarchy_dirs.extend(
            state_dir / "hierarchy"
            for state_dir in states_dir.iterdir()
            if state_dir.is_dir() and not state_dir.name.startswith(".")
        )
    hierarchy_dirs.sort()
    for hierarchy_dir in hierarchy_dirs:
        if not (hierarchy_dir / "unit_metrics.csv").is_file():
            continue
        old_manifest = hierarchy_dir / "unit_workflow_manifest.json"
        payload = _read_json(old_manifest) if old_manifest.is_file() else {}
        state_dir = hierarchy_dir.parent
        manifest_links = Path(str(payload["input_links_path"])).expanduser() if payload.get("input_links_path") else None
        manifest_nodes = Path(str(payload["input_nodes_path"])).expanduser() if payload.get("input_nodes_path") else None
        links = manifest_links if manifest_links and not manifest_links.name.startswith(".") and manifest_links.is_file() else _first_existing([
            *sorted((state_dir / "variant" / "widths").glob("*links_with_width_families.gpkg")),
            *sorted((state_dir / "variant" / "directed").glob("*_directed_links.gpkg")),
        ])
        nodes = manifest_nodes if manifest_nodes and not manifest_nodes.name.startswith(".") and manifest_nodes.is_file() else _first_existing(
            sorted((state_dir / "variant" / "directed").glob("*_directed_nodes.gpkg"))
        )
        reason = "final directed links/nodes and width product available" if links and nodes else "missing retained final links or nodes"
        targets.append(RetrofitTarget(hierarchy_dir, links, nodes, old_manifest if old_manifest.is_file() else None, bool(links and nodes), reason))
    return targets


def build_dry_run_manifest(output_root: str | Path, targets: Sequence[RetrofitTarget] | None = None) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    targets = list(targets) if targets is not None else discover_targets(root)
    entries = []
    for target in targets:
        old_files = [name for name in HIERARCHY_FILES if (target.hierarchy_dir / name).is_file()]
        entries.append({
            "hierarchy_dir": str(target.hierarchy_dir), "state_id": target.hierarchy_dir.parent.name,
            "invalid_artifacts": old_files,
            "excluded_artifacts_retained": [name for name in ("bubble_equivalent_metrics.csv", "bubble_edge_resistance.csv", "bubble_node_heads.csv") if (target.hierarchy_dir / name).is_file()],
            "replacement_action": "write versioned sidecar; retain originals unchanged",
            "replacement_dir": str(target.hierarchy_dir / "backfills" / RETROFIT_VERSION),
            "source_links": str(target.links_path) if target.links_path else None,
            "source_nodes": str(target.nodes_path) if target.nodes_path else None,
            "safe_to_recompute": target.safe_to_recompute, "reason": target.reason,
        })
    return {"schema_version": "hierarchy-length-retrofit-manifest-v1", "created_at": _utc_now(), "retrofit_version": RETROFIT_VERSION, "policy": POLICY, "dry_run": True, "n_targets": len(entries), "n_safe": sum(item["safe_to_recompute"] for item in entries), "targets": entries}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result_frames(results: Any) -> dict[str, pd.DataFrame]:
    return {
        "unit_summary.csv": results.unit_summary, "path_metrics.csv": results.path_metrics,
        "unit_metrics.csv": results.unit_metrics, "hierarchy_level_metrics.csv": results.hierarchy_level_metrics,
        "collapse_ranking.csv": results.collapse_ranking, "merge_tree.csv": results.merge_tree,
        "ordered_group_partitions.csv": results.merge_tree, "group_count_summary.csv": results.group_count_summary,
        "bubble_summary.csv": results.bubble_summary, "selected_groups.csv": results.selected_groups,
    }


def _changed_fields(old_path: Path, new_frame: pd.DataFrame, name: str) -> list[str]:
    if not old_path.is_file():
        return list(new_frame.columns)
    old = pd.read_csv(old_path)
    keys = [key for key in KEY_COLUMNS.get(name, ()) if key in old and key in new_frame]
    if not keys:
        return sorted(set(old.columns).symmetric_difference(new_frame.columns))
    merged = old.merge(new_frame, on=keys, how="outer", suffixes=("__old", "__new"), indicator=True)
    changed: list[str] = []
    for column in sorted((set(old.columns) & set(new_frame.columns)) - set(keys)):
        left, right = merged[f"{column}__old"], merged[f"{column}__new"]
        numeric = pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
        unequal = (~left.eq(right)) & ~(left.isna() & right.isna()) if not numeric else ~pd.Series(pd.to_numeric(left, errors="coerce")).round(12).eq(pd.Series(pd.to_numeric(right, errors="coerce")).round(12)) & ~(left.isna() & right.isna())
        if bool(unequal.any()) or bool((merged["_merge"] != "both").any()):
            changed.append(column)
    return changed


def execute_retrofit(output_root: str | Path, targets: Sequence[RetrofitTarget] | None = None) -> dict[str, Any]:
    """Recompute safe hierarchy targets into versioned sidecars only."""
    root = Path(output_root).expanduser().resolve()
    targets = list(targets) if targets is not None else discover_targets(root)
    execution: list[dict[str, Any]] = []
    for target in targets:
        entry: dict[str, Any] = {"hierarchy_dir": str(target.hierarchy_dir), "state_id": target.hierarchy_dir.parent.name, "safe_to_recompute": target.safe_to_recompute}
        if not target.safe_to_recompute:
            entry["status"] = "skipped"; entry["reason"] = target.reason; execution.append(entry); continue
        destination = target.hierarchy_dir / "backfills" / RETROFIT_VERSION
        existing_provenance = destination / "provenance.json"
        if existing_provenance.is_file():
            previous = _read_json(existing_provenance)
            if (
                previous.get("retrofit_version") == RETROFIT_VERSION
                and previous.get("source_links") == str(target.links_path)
                and previous.get("source_nodes") == str(target.nodes_path)
            ):
                entry.update({"status": "already_written", "replacement_dir": str(destination)})
                execution.append(entry)
                continue
        try:
            results = run_unit_workflow(target.links_path, target.nodes_path)
        except Exception as exc:  # retain other audited states if one legacy product is unreadable
            entry.update({
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            execution.append(entry)
            continue
        destination.mkdir(parents=True, exist_ok=True)
        changes: dict[str, list[str]] = {}
        sources: dict[str, str] = {}
        for name, frame in _result_frames(results).items():
            old = target.hierarchy_dir / name
            changes[name] = _changed_fields(old, frame, name)
            if old.is_file(): sources[name] = _sha256(old)
            frame.to_csv(destination / name, index=False)
        provenance = {"schema_version": "hierarchy-length-retrofit-execution-v1", "executed_at": _utc_now(), "retrofit_version": RETROFIT_VERSION, "policy": POLICY, "source_links": str(target.links_path), "source_nodes": str(target.nodes_path), "source_artifact_sha256": sources, "changed_fields": changes, "original_hierarchy_dir": str(target.hierarchy_dir), "equivalent_resistance": "retained; not recomputed", "rapid": "retained; not rerun"}
        _write_json(destination / "provenance.json", provenance)
        entry.update({"status": "written", "replacement_dir": str(destination), "changed_fields": changes})
        execution.append(entry)
    payload = {"schema_version": "hierarchy-length-retrofit-execution-v1", "executed_at": _utc_now(), "retrofit_version": RETROFIT_VERSION, "policy": POLICY, "n_targets": len(execution), "n_written": sum(row.get("status") == "written" for row in execution), "n_already_written": sum(row.get("status") == "already_written" for row in execution), "n_failed": sum(row.get("status") == "failed" for row in execution), "targets": execution}
    _write_json(root / "hierarchy_length_retrofit_execution_manifest.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit and backfill hierarchy metrics using physical len before legacy len_adj.")
    parser.add_argument("--output-root", default="network_variants/output")
    parser.add_argument("--execute", action="store_true", help="Write versioned hierarchy sidecars. Default is dry-run only.")
    parser.add_argument("--variant", action="append", default=[], help="Limit to output directory names, e.g. 56_compound_bubbles__anchored.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    targets = discover_targets(root)
    if args.variant:
        allowed = set(args.variant)
        targets = [target for target in targets if any(part in allowed for part in target.hierarchy_dir.parts)]
    dry_run = build_dry_run_manifest(root, targets)
    _write_json(root / "hierarchy_length_retrofit_dry_run_manifest.json", dry_run)
    if not args.execute:
        print(json.dumps({"dry_run_manifest": str(root / "hierarchy_length_retrofit_dry_run_manifest.json"), "n_targets": dry_run["n_targets"], "n_safe": dry_run["n_safe"]}, indent=2)); return 0
    result = execute_retrofit(root, targets)
    print(json.dumps({"execution_manifest": str(root / "hierarchy_length_retrofit_execution_manifest.json"), "n_written": result["n_written"]}, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
