from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import geopandas as gpd
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
MISSING_STATUS = "missing"
CORRUPT_STATUS = "corrupt_archive"
STALE_MASK_STATUS = "stale_mask"
STALE_TOPOLOGY_STATUS = "stale_topology"
STALE_LINK_IDS_STATUS = "stale_link_ids"
STALE_WIDTH_METHOD_STATUS = "stale_width_method"


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
    manifest["review_status"] = CURRENT_STATUS
    manifest.pop("notes", None)
    _atomic_write_json(manifest, manifest_path)
    return {
        "example_id": str(manifest["example_id"]),
        "archive_dir": str(archive),
        "manifest": str(manifest_path),
        "status": CURRENT_STATUS,
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
        status = CURRENT_STATUS
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
