from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
import networkx as nx
import pandas as pd
import rasterio

from network_variants.width_review_archive import (
    ARCHIVE_MANIFEST_NAME,
    _atomic_write_json,
    _sha256,
)


FINGERPRINT_SCHEMA_VERSION = "width-review-fingerprint-v1"
WIDTH_REVIEW_INPUT_VERSION = "two-bank-width-review-input-v1"
CURRENT_STATUS = "current"
INCOMPLETE_REVIEW_STATUS = "incomplete_review"
MISSING_STATUS = "missing"
CORRUPT_STATUS = "corrupt_archive"
STALE_MASK_STATUS = "stale_mask"
STALE_TOPOLOGY_STATUS = "stale_topology"
STALE_LINK_IDS_STATUS = "stale_link_ids"
STALE_WIDTH_METHOD_STATUS = "stale_width_method"
DIRECTION_ONLY_CURRENT_STATUS = "direction_only_current"
DIRECTION_CONTINUITY_SCHEMA_VERSION = "width-review-direction-continuity-v1"


def _json_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _crs_text(frame: gpd.GeoDataFrame) -> str | None:
    return frame.crs.to_string() if frame.crs is not None else None


def _geometry_hex(geometry: object) -> str | None:
    if geometry is None or getattr(geometry, "is_empty", False):
        return None
    return str(getattr(geometry, "wkb_hex"))


def _geometry_sha256(geometry: object) -> str | None:
    geometry_hex = _geometry_hex(geometry)
    return hashlib.sha256(geometry_hex.encode("ascii")).hexdigest() if geometry_hex is not None else None


def fingerprint_mask(mask_path: str | Path) -> dict[str, Any]:
    path = Path(mask_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reviewed mask does not exist: {path}")
    with rasterio.open(path) as source:
        profile = {
            "height": int(source.height),
            "width": int(source.width),
            "count": int(source.count),
            "dtypes": list(source.dtypes),
            "crs": source.crs.to_string() if source.crs is not None else None,
            "transform": [float(value) for value in source.transform],
            "nodata": source.nodata,
        }
    return {
        "path": str(path),
        "file_size": int(path.stat().st_size),
        "file_sha256": _sha256(path),
        "profile": profile,
        "profile_sha256": _json_digest(profile),
    }


def fingerprint_network(
    links_path: str | Path,
    nodes_path: str | Path,
) -> dict[str, Any]:
    links_file = Path(links_path).expanduser().resolve()
    nodes_file = Path(nodes_path).expanduser().resolve()
    if not links_file.exists():
        raise FileNotFoundError(f"Reviewed links do not exist: {links_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Reviewed nodes do not exist: {nodes_file}")
    links = gpd.read_file(links_file)
    nodes = gpd.read_file(nodes_file)
    required_link_fields = {"id_link", "id_us_node", "id_ds_node"}
    required_node_fields = {"id_node"}
    if not required_link_fields.issubset(links.columns):
        raise ValueError(f"Reviewed links are missing fields: {sorted(required_link_fields.difference(links.columns))}")
    if not required_node_fields.issubset(nodes.columns):
        raise ValueError(f"Reviewed nodes are missing fields: {sorted(required_node_fields.difference(nodes.columns))}")

    link_records: list[dict[str, Any]] = []
    for row in links.itertuples(index=False):
        link_records.append(
            {
                "id_link": int(row.id_link),
                "id_us_node": int(row.id_us_node),
                "id_ds_node": int(row.id_ds_node),
                "geometry_sha256": _geometry_sha256(row.geometry),
            }
        )
    link_records.sort(key=lambda record: record["id_link"])
    topology_records = sorted(
        (
            {
                "id_us_node": record["id_us_node"],
                "id_ds_node": record["id_ds_node"],
                "geometry_sha256": record["geometry_sha256"],
            }
            for record in link_records
        ),
        key=lambda record: (
            record["id_us_node"],
            record["id_ds_node"],
            record["geometry_sha256"] or "",
        ),
    )

    node_records: list[dict[str, Any]] = []
    for row in nodes.itertuples(index=False):
        record = {
            "id_node": int(row.id_node),
            "geometry_sha256": _geometry_sha256(row.geometry),
        }
        for field in ("is_inlet", "is_outlet"):
            if field in nodes.columns:
                value = getattr(row, field)
                record[field] = None if pd.isna(value) else bool(value)
        node_records.append(record)
    node_records.sort(key=lambda record: record["id_node"])

    crs = {"links": _crs_text(links), "nodes": _crs_text(nodes)}
    return {
        "links_path": str(links_file),
        "nodes_path": str(nodes_file),
        "link_count": int(len(links)),
        "node_count": int(len(nodes)),
        "crs": crs,
        "link_ids_sha256": _json_digest([record["id_link"] for record in link_records]),
        "link_identity_sha256": _json_digest({"crs": crs["links"], "records": link_records}),
        "link_topology_sha256": _json_digest({"crs": crs["links"], "records": topology_records}),
        "node_identity_sha256": _json_digest({"crs": crs["nodes"], "records": node_records}),
        "link_identity_records": link_records,
        "node_identity_records": node_records,
    }


def propose_link_id_remap(
    expected_records: Sequence[Mapping[str, Any]],
    current_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Match old and new IDs only where directed endpoints and geometry agree."""
    def signature(record: Mapping[str, Any]) -> tuple[int, int, str | None]:
        return (
            int(record["id_us_node"]),
            int(record["id_ds_node"]),
            record.get("geometry_sha256"),
        )

    expected_by_signature: dict[tuple[int, int, str | None], list[int]] = {}
    current_by_signature: dict[tuple[int, int, str | None], list[int]] = {}
    for record in expected_records:
        expected_by_signature.setdefault(signature(record), []).append(int(record["id_link"]))
    for record in current_records:
        current_by_signature.setdefault(signature(record), []).append(int(record["id_link"]))

    mapping: list[dict[str, int]] = []
    ambiguities: list[dict[str, Any]] = []
    for key in sorted(set(expected_by_signature).union(current_by_signature), key=str):
        old_ids = sorted(expected_by_signature.get(key, []))
        new_ids = sorted(current_by_signature.get(key, []))
        if len(old_ids) == 1 and len(new_ids) == 1:
            mapping.append({"reviewed_id_link": old_ids[0], "current_id_link": new_ids[0]})
        else:
            ambiguities.append(
                {
                    "id_us_node": key[0],
                    "id_ds_node": key[1],
                    "geometry_sha256": key[2],
                    "reviewed_id_links": old_ids,
                    "current_id_links": new_ids,
                }
            )
    changed = [row for row in mapping if row["reviewed_id_link"] != row["current_id_link"]]
    return {
        "status": "complete" if not ambiguities else "ambiguous",
        "requires_confirmation": True,
        "matched_link_count": int(len(mapping)),
        "changed_link_count": int(len(changed)),
        "mapping": mapping,
        "ambiguities": ambiguities,
    }


def _resolve_current_paths(
    archive_dir: Path,
    manifest: Mapping[str, Any],
    *,
    mask_path: str | Path | None = None,
    links_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    example_id = str(manifest.get("example_id", archive_dir.parent.name))
    run_dir = archive_dir.parent
    resolved_links = (
        Path(links_path).expanduser().resolve()
        if links_path is not None
        else run_dir / "rivgraph" / f"{example_id}_links.gpkg"
    )
    resolved_nodes = (
        Path(nodes_path).expanduser().resolve()
        if nodes_path is not None
        else run_dir / "rivgraph" / f"{example_id}_nodes.gpkg"
    )
    resolved_mask = (
        Path(mask_path).expanduser().resolve()
        if mask_path is not None
        else run_dir / "masks_cleaned" / f"{example_id}_cleaned.tif"
    )
    return resolved_mask, resolved_links, resolved_nodes


def _archive_integrity_issues(archive_dir: Path, manifest: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    files = manifest.get("files", {}) or {}
    expected_hashes = manifest.get("archive_file_sha256", {}) or {}
    for key, expected_hash in expected_hashes.items():
        filename = files.get(key)
        if not filename:
            issues.append(f"archive manifest has no filename for {key}")
            continue
        path = archive_dir / str(filename)
        if not path.exists():
            issues.append(f"archive file is missing: {path.name}")
            continue
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            issues.append(f"archive file hash changed: {path.name}")
    return issues


def upgrade_review_fingerprints(
    archive_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
    links_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
    width_method_version: str = WIDTH_REVIEW_INPUT_VERSION,
) -> dict[str, Any]:
    archive = Path(archive_dir).expanduser().resolve()
    manifest_path = archive / ARCHIVE_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Width-review manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity_issues = _archive_integrity_issues(archive, manifest)
    if integrity_issues:
        raise ValueError("Cannot fingerprint a damaged review archive: " + "; ".join(integrity_issues))
    current_mask, current_links, current_nodes = _resolve_current_paths(
        archive,
        manifest,
        mask_path=mask_path,
        links_path=links_path,
        nodes_path=nodes_path,
    )
    manifest["fingerprint_schema_version"] = FINGERPRINT_SCHEMA_VERSION
    manifest["fingerprinted_at"] = datetime.now(timezone.utc).isoformat()
    manifest["width_method_version"] = str(width_method_version)
    manifest["fingerprints"] = {
        "mask": fingerprint_mask(current_mask),
        "network": fingerprint_network(current_links, current_nodes),
    }
    completion_status = str(manifest.get("review_completion_status", "complete"))
    review_status = INCOMPLETE_REVIEW_STATUS if completion_status == "incomplete" else CURRENT_STATUS
    manifest["review_status"] = review_status
    manifest.pop("notes", None)
    _atomic_write_json(manifest, manifest_path)
    return {
        "example_id": str(manifest["example_id"]),
        "archive_dir": str(archive),
        "manifest": str(manifest_path),
        "status": review_status,
    }


def validate_review_fingerprints(
    archive_dir: str | Path,
    *,
    mask_path: str | Path | None = None,
    links_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
    width_method_version: str = WIDTH_REVIEW_INPUT_VERSION,
) -> dict[str, Any]:
    archive = Path(archive_dir).expanduser().resolve()
    manifest_path = archive / ARCHIVE_MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "archive_dir": str(archive),
            "status": MISSING_STATUS,
            "issues": [f"review manifest is missing: {manifest_path}"],
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    issues = _archive_integrity_issues(archive, manifest)
    if issues:
        return {
            "example_id": str(manifest.get("example_id", archive.parent.name)),
            "archive_dir": str(archive),
            "status": CORRUPT_STATUS,
            "issues": issues,
        }
    expected = manifest.get("fingerprints")
    if not expected or manifest.get("fingerprint_schema_version") != FINGERPRINT_SCHEMA_VERSION:
        return {
            "example_id": str(manifest.get("example_id", archive.parent.name)),
            "archive_dir": str(archive),
            "status": MISSING_STATUS,
            "issues": ["review archive has not been upgraded with current fingerprints"],
        }

    current_mask, current_links, current_nodes = _resolve_current_paths(
        archive,
        manifest,
        mask_path=mask_path,
        links_path=links_path,
        nodes_path=nodes_path,
    )
    try:
        current = {
            "mask": fingerprint_mask(current_mask),
            "network": fingerprint_network(current_links, current_nodes),
        }
    except (FileNotFoundError, ValueError) as exc:
        return {
            "example_id": str(manifest.get("example_id", archive.parent.name)),
            "archive_dir": str(archive),
            "status": MISSING_STATUS,
            "issues": [str(exc)],
        }

    stale_mask = (
        current["mask"]["file_sha256"] != expected["mask"]["file_sha256"]
        or current["mask"]["profile_sha256"] != expected["mask"]["profile_sha256"]
    )
    expected_network = expected["network"]
    current_network = current["network"]
    node_changed = current_network["node_identity_sha256"] != expected_network["node_identity_sha256"]
    link_identity_changed = current_network["link_identity_sha256"] != expected_network["link_identity_sha256"]
    link_topology_changed = current_network["link_topology_sha256"] != expected_network["link_topology_sha256"]
    id_only_change = link_identity_changed and not link_topology_changed and not node_changed
    topology_change = node_changed or link_topology_changed
    stale_width_method = str(manifest.get("width_method_version")) != str(width_method_version)

    reasons: list[str] = []
    if stale_mask:
        reasons.append("cleaned mask fingerprint changed")
    if topology_change:
        reasons.append("network topology, direction, node identity, or geometry changed")
    elif id_only_change:
        reasons.append("link identities changed while the topology/geometry multiset remained equal")
    if stale_width_method:
        reasons.append(
            f"width method changed from {manifest.get('width_method_version')} to {width_method_version}"
        )

    if stale_mask:
        status = STALE_MASK_STATUS
    elif topology_change:
        status = STALE_TOPOLOGY_STATUS
    elif id_only_change:
        status = STALE_LINK_IDS_STATUS
    elif stale_width_method:
        status = STALE_WIDTH_METHOD_STATUS
    else:
        completion_status = str(manifest.get("review_completion_status", "complete"))
        status = INCOMPLETE_REVIEW_STATUS if completion_status == "incomplete" else CURRENT_STATUS
    result = {
        "example_id": str(manifest.get("example_id", archive.parent.name)),
        "archive_dir": str(archive),
        "status": status,
        "issues": reasons,
        "expected": expected,
        "current": current,
    }
    if id_only_change:
        result["link_id_remap_proposal"] = propose_link_id_remap(
            expected_network.get("link_identity_records", []),
            current_network.get("link_identity_records", []),
        )
    return result


def _direction_only_network_audit(
    expected_network: Mapping[str, Any],
    *,
    expected_links_path: str | Path,
    current_links_path: str | Path,
    current_nodes_path: str | Path,
) -> dict[str, Any]:
    """Verify an identity-preserving graph differs only by link orientation.

    This deliberately compares geometries with ``equals`` instead of WKB
    hashes: a direction correction is allowed to reverse a LineString's point
    order, but it may not move, split, merge, add, or remove a link/node.
    """
    expected_links = list(expected_network.get("link_identity_records", []))
    expected_nodes = list(expected_network.get("node_identity_records", []))
    if not expected_links or not expected_nodes:
        raise ValueError("The archived review fingerprint has no link/node identity records.")

    snapshot_records = {int(record["id_link"]): record for record in expected_links}
    expected_links_frame = gpd.read_file(Path(expected_links_path).expanduser().resolve())
    if "id_link" not in expected_links_frame.columns:
        raise ValueError("Archived reviewed-link snapshot is missing id_link.")
    expected_links_frame["id_link"] = pd.to_numeric(
        expected_links_frame["id_link"], errors="raise"
    ).astype(int)
    if expected_links_frame["id_link"].duplicated().any():
        raise ValueError("Archived reviewed-link snapshot has duplicate id_link values.")
    expected_link_geometry = expected_links_frame.set_index("id_link", drop=False).geometry
    current_links = gpd.read_file(Path(current_links_path).expanduser().resolve())
    current_nodes = gpd.read_file(Path(current_nodes_path).expanduser().resolve())
    required_links = {"id_link", "id_us_node", "id_ds_node"}
    if not required_links.issubset(current_links.columns) or "id_node" not in current_nodes.columns:
        raise ValueError("Current graph is missing required link/node identity fields.")
    current_links["id_link"] = pd.to_numeric(current_links["id_link"], errors="raise").astype(int)
    if current_links["id_link"].duplicated().any():
        raise ValueError("Current graph has duplicate id_link values.")
    current_by_id = current_links.set_index("id_link", drop=False)

    issues: list[str] = []
    if set(snapshot_records) != set(current_by_id.index.astype(int)):
        issues.append("stable link-ID set changed")
    unchanged: list[int] = []
    reversed_ids: list[int] = []
    changed_non_directional: list[int] = []
    changed_link_geometries: list[int] = []
    for link_id in sorted(set(snapshot_records).intersection(current_by_id.index.astype(int))):
        previous = snapshot_records[int(link_id)]
        current = current_by_id.loc[int(link_id)]
        previous_nodes = (int(previous["id_us_node"]), int(previous["id_ds_node"]))
        current_nodes_pair = (int(current.id_us_node), int(current.id_ds_node))
        # Snapshot WKB is the durable record used in the ordinary fingerprint.
        # Read it only if necessary: the caller may provide an archive whose
        # historical source file is no longer available.
        if sorted(previous_nodes) != sorted(current_nodes_pair):
            changed_non_directional.append(int(link_id))
            continue
        if not expected_link_geometry.loc[int(link_id)].equals(current.geometry):
            changed_link_geometries.append(int(link_id))
            continue
        if previous_nodes == current_nodes_pair:
            unchanged.append(int(link_id))
        elif previous_nodes == current_nodes_pair[::-1]:
            reversed_ids.append(int(link_id))
        else:  # Defensive: this should be impossible after the unordered test.
            changed_non_directional.append(int(link_id))

    # Node geometry/identity must be exact except for inlet/outlet flags, which
    # necessarily change for a whole-network reversal.
    expected_node_by_id = {int(record["id_node"]): record for record in expected_nodes}
    current_nodes["id_node"] = pd.to_numeric(current_nodes["id_node"], errors="raise").astype(int)
    if current_nodes["id_node"].duplicated().any():
        issues.append("current graph has duplicate id_node values")
    current_node_by_id = current_nodes.set_index("id_node", drop=False)
    if set(expected_node_by_id) != set(current_node_by_id.index.astype(int)):
        issues.append("stable node-ID set changed")
    changed_node_geometries: list[int] = []
    for node_id in sorted(set(expected_node_by_id).intersection(current_node_by_id.index.astype(int))):
        expected_geometry = expected_node_by_id[int(node_id)].get("geometry_sha256")
        current_geometry = _geometry_sha256(current_node_by_id.loc[int(node_id)].geometry)
        if expected_geometry != current_geometry:
            changed_node_geometries.append(int(node_id))
    if changed_node_geometries:
        issues.append("node geometries changed")
    if changed_non_directional:
        issues.append("one or more links changed endpoints beyond a pure reversal")
    if changed_link_geometries:
        issues.append("one or more link geometries changed")
    if not reversed_ids:
        issues.append("no reversed links were found; this is not a direction-continuity correction")

    # Geometry/identity continuity alone is insufficient: RivGraph can retain
    # a small number of locally ``certain`` directions after a whole-network
    # outlet reversal. Those retained links can close a directed loop even
    # though every link still has the same undirected endpoints. Downstream
    # hierarchy and equivalent-resistance work require a DAG, so make that
    # requirement part of the direction-continuity proof itself.
    directed = nx.DiGraph()
    for row in current_links.itertuples(index=False):
        directed.add_edge(
            int(row.id_us_node),
            int(row.id_ds_node),
            id_link=int(row.id_link),
        )
    is_dag = nx.is_directed_acyclic_graph(directed)
    directed_cycles: list[dict[str, list[int]]] = []
    if not is_dag:
        for cycle_nodes in nx.simple_cycles(directed, length_bound=32):
            cycle_links = [
                int(directed[upstream][downstream]["id_link"])
                for upstream, downstream in zip(cycle_nodes, cycle_nodes[1:] + cycle_nodes[:1])
            ]
            directed_cycles.append(
                {"node_ids": [int(node_id) for node_id in cycle_nodes], "link_ids": cycle_links}
            )
            if len(directed_cycles) >= 10:
                break
        issues.append("corrected graph contains directed cycle(s)")

    return {
        "status": "valid" if not issues else "invalid",
        "issues": issues,
        "link_count": int(len(current_links)),
        "node_count": int(len(current_nodes)),
        "unchanged_link_ids": unchanged,
        "reversed_link_ids": reversed_ids,
        "non_directional_changed_link_ids": changed_non_directional,
        "changed_link_geometry_ids": changed_link_geometries,
        "changed_node_geometry_ids": changed_node_geometries,
        "is_directed_acyclic": bool(is_dag),
        "directed_cycles": directed_cycles,
    }


def write_direction_only_review_continuity(
    archive_dir: str | Path,
    *,
    mask_path: str | Path,
    links_path: str | Path,
    nodes_path: str | Path,
    prior_exit_sides: str,
    corrected_exit_sides: str,
    output_path: str | Path | None = None,
    rationale: str = "Validated whole-network direction correction; geometry and stable IDs are unchanged.",
) -> dict[str, Any]:
    """Write an explicit approval record for a pure direction correction.

    This function is intentionally separate from the ordinary fingerprint
    upgrade path. A stale archive can be accepted only when an operator calls
    this function and it independently proves that the current graph differs
    solely by reversed directed endpoints.
    """
    archive = Path(archive_dir).expanduser().resolve()
    manifest_path = archive / ARCHIVE_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Width-review manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity_issues = _archive_integrity_issues(archive, manifest)
    if integrity_issues:
        raise ValueError("Cannot create continuity record for a damaged review archive: " + "; ".join(integrity_issues))
    expected = manifest.get("fingerprints")
    if not expected or manifest.get("fingerprint_schema_version") != FINGERPRINT_SCHEMA_VERSION:
        raise ValueError("Review archive needs current fingerprints before direction continuity can be approved.")
    current_mask = fingerprint_mask(mask_path)
    expected_mask = expected["mask"]
    if (
        current_mask["file_sha256"] != expected_mask["file_sha256"]
        or current_mask["profile_sha256"] != expected_mask["profile_sha256"]
    ):
        raise ValueError("Direction continuity is invalid because the cleaned mask changed.")
    audit = _direction_only_network_audit(
        expected["network"],
        expected_links_path=archive / str((manifest.get("files", {}) or {}).get("links_snapshot", "reviewed_links_snapshot.gpkg")),
        current_links_path=links_path,
        current_nodes_path=nodes_path,
    )
    if audit["status"] != "valid":
        raise ValueError("Direction continuity is invalid: " + "; ".join(audit["issues"]))
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else archive / "direction_only_continuity.json"
    )
    payload = {
        "schema_version": DIRECTION_CONTINUITY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_dir": str(archive),
        "archive_manifest": str(manifest_path),
        "archive_manifest_sha256": _sha256(manifest_path),
        "example_id": str(manifest.get("example_id", archive.parent.name)),
        "prior_exit_sides": str(prior_exit_sides),
        "corrected_exit_sides": str(corrected_exit_sides),
        "mask": current_mask,
        "current_links_path": str(Path(links_path).expanduser().resolve()),
        "current_nodes_path": str(Path(nodes_path).expanduser().resolve()),
        "rationale": str(rationale),
        "audit": audit,
    }
    _atomic_write_json(payload, destination)
    return {**payload, "continuity_path": str(destination)}


def validate_direction_only_review_continuity(
    archive_dir: str | Path,
    *,
    continuity_path: str | Path,
    mask_path: str | Path,
    links_path: str | Path,
    nodes_path: str | Path,
    width_method_version: str = WIDTH_REVIEW_INPUT_VERSION,
) -> dict[str, Any]:
    """Validate a previously written, narrow direction-only approval record."""
    ordinary = validate_review_fingerprints(
        archive_dir,
        mask_path=mask_path,
        links_path=links_path,
        nodes_path=nodes_path,
        width_method_version=width_method_version,
    )
    if ordinary.get("status") == CURRENT_STATUS:
        return ordinary
    continuity_file = Path(continuity_path).expanduser().resolve()
    if not continuity_file.exists():
        return ordinary
    try:
        record = json.loads(continuity_file.read_text(encoding="utf-8"))
        archive = Path(archive_dir).expanduser().resolve()
        manifest_path = archive / ARCHIVE_MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != DIRECTION_CONTINUITY_SCHEMA_VERSION:
            raise ValueError("continuity schema version is not recognised")
        if str(record.get("archive_manifest_sha256")) != _sha256(manifest_path):
            raise ValueError("archived review manifest changed after continuity approval")
        if str(record.get("archive_dir")) != str(archive):
            raise ValueError("continuity record references a different review archive")
        if str(manifest.get("width_method_version")) != str(width_method_version):
            raise ValueError("width method changed after continuity approval")
        expected = manifest["fingerprints"]
        current_mask = fingerprint_mask(mask_path)
        if (
            current_mask["file_sha256"] != expected["mask"]["file_sha256"]
            or current_mask["profile_sha256"] != expected["mask"]["profile_sha256"]
        ):
            raise ValueError("cleaned mask fingerprint changed")
        audit = _direction_only_network_audit(
            expected["network"],
            expected_links_path=archive / str((manifest.get("files", {}) or {}).get("links_snapshot", "reviewed_links_snapshot.gpkg")),
            current_links_path=links_path,
            current_nodes_path=nodes_path,
        )
        if audit["status"] != "valid":
            raise ValueError("; ".join(audit["issues"]))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            **ordinary,
            "direction_continuity_path": str(continuity_file),
            "direction_continuity_issues": [str(exc)],
        }
    return {
        "example_id": str(manifest.get("example_id", archive.parent.name)),
        "archive_dir": str(archive),
        "status": DIRECTION_ONLY_CURRENT_STATUS,
        "issues": [],
        "expected": expected,
        "current": {
            "mask": current_mask,
            "network": fingerprint_network(links_path, nodes_path),
        },
        "direction_continuity_path": str(continuity_file),
        "direction_continuity": record,
        "direction_continuity_audit": audit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or validate width-review freshness fingerprints.")
    parser.add_argument("action", choices=("upgrade", "validate"))
    parser.add_argument("archive_dir", nargs="+", help="One or more permanent width_review directories.")
    parser.add_argument("--width-method-version", default=WIDTH_REVIEW_INPUT_VERSION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    operation = upgrade_review_fingerprints if args.action == "upgrade" else validate_review_fingerprints
    results = [
        operation(path, width_method_version=args.width_method_version)
        for path in args.archive_dir
    ]
    print(json.dumps(results, indent=2))
    return 0 if all(result["status"] == CURRENT_STATUS for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
