from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd

from network_variants.width_review_archive import (
    ARCHIVE_DECISIONS_NAME,
    ARCHIVE_LINKS_SNAPSHOT_NAME,
    ARCHIVE_MANIFEST_NAME,
    _atomic_write_csv,
    _atomic_write_json,
    _sha256,
)


MIGRATION_SCHEMA_VERSION = "width-review-direction-migration-v1"
DEFAULT_MIGRATION_DIR_NAME = "width_review_direction_migration"
DEFAULT_SEED_NAME = "migrated_width_review_seed.csv"
DEFAULT_MANIFEST_NAME = "width_review_direction_migration_manifest.json"


def _integer_links(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    required = {"id_link", "id_us_node", "id_ds_node"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Link file is missing required columns: {sorted(missing)}")
    result = frame.copy()
    for column in sorted(required):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    if result[list(required)].isna().any().any():
        raise ValueError("Link IDs and endpoint node IDs must be complete integers.")
    for column in sorted(required):
        result[column] = result[column].astype(int)
    if result["id_link"].duplicated().any():
        raise ValueError("Link IDs must be unique before width-review migration.")
    return result


def derive_direction_affected_context(
    old_links: gpd.GeoDataFrame,
    new_links: gpd.GeoDataFrame,
    *,
    changed_link_ids: Iterable[int],
    candidate_link_ids: Iterable[int],
) -> dict[str, list[int]]:
    """Validate a direction-only edit and return its junction-local review impact."""
    old = _integer_links(old_links).set_index("id_link", drop=False)
    new = _integer_links(new_links).set_index("id_link", drop=False)
    if set(old.index) != set(new.index):
        raise ValueError("Cannot migrate: link IDs changed between the archived and current base.")

    requested = sorted({int(value) for value in changed_link_ids})
    if not requested:
        raise ValueError("At least one --changed-link-id is required.")
    missing = sorted(set(requested).difference(old.index))
    if missing:
        raise ValueError(f"Changed link IDs are absent from the network: {missing}")

    actual_direction_changes: list[int] = []
    for link_id in sorted(old.index):
        old_pair = (int(old.at[link_id, "id_us_node"]), int(old.at[link_id, "id_ds_node"]))
        new_pair = (int(new.at[link_id, "id_us_node"]), int(new.at[link_id, "id_ds_node"]))
        if old_pair != new_pair:
            actual_direction_changes.append(int(link_id))
    if actual_direction_changes != requested:
        raise ValueError(
            "Current topology changes do not exactly match the requested direction changes: "
            f"requested={requested}, actual={actual_direction_changes}."
        )

    affected_nodes: set[int] = set()
    for link_id in requested:
        old_pair = (int(old.at[link_id, "id_us_node"]), int(old.at[link_id, "id_ds_node"]))
        new_pair = (int(new.at[link_id, "id_us_node"]), int(new.at[link_id, "id_ds_node"]))
        if old_pair != tuple(reversed(new_pair)):
            raise ValueError(
                f"Link {link_id} is not a pure direction reversal: old={old_pair}, new={new_pair}."
            )
        old_geom = old.at[link_id, "geometry"]
        new_geom = new.at[link_id, "geometry"]
        if old_geom is None or new_geom is None or not old_geom.equals(new_geom):
            raise ValueError(f"Link {link_id} geometry changed; this is not a direction-only migration.")
        affected_nodes.update(old_pair)
        affected_nodes.update(new_pair)

    incident_mask = (
        old["id_us_node"].isin(affected_nodes)
        | old["id_ds_node"].isin(affected_nodes)
        | new["id_us_node"].isin(affected_nodes)
        | new["id_ds_node"].isin(affected_nodes)
    )
    incident = sorted(int(value) for value in old.loc[incident_mask, "id_link"])
    candidates = {int(value) for value in candidate_link_ids}
    return {
        "changed_link_ids": requested,
        "affected_node_ids": sorted(affected_nodes),
        "incident_link_ids": incident,
        "affected_candidate_link_ids": sorted(candidates.intersection(incident)),
    }


def build_migrated_decision_seed(
    old_decisions: pd.DataFrame,
    *,
    candidate_link_ids: Iterable[int],
    affected_candidate_link_ids: Iterable[int],
) -> pd.DataFrame:
    """Carry decisions by stable ID, clearing every direction-affected candidate."""
    required = {"id_link", "review_decision"}
    missing = required.difference(old_decisions.columns)
    if missing:
        raise ValueError(f"Archived decisions are missing columns: {sorted(missing)}")
    old = old_decisions.loc[:, ["id_link", "review_decision"]].copy()
    old["id_link"] = pd.to_numeric(old["id_link"], errors="coerce").astype("Int64")
    old = old.dropna(subset=["id_link"]).copy()
    old["id_link"] = old["id_link"].astype(int)
    old["review_decision"] = old["review_decision"].astype("string").str.strip().str.lower()
    old.loc[~old["review_decision"].isin(["good", "bad"]), "review_decision"] = pd.NA
    if old["id_link"].duplicated().any():
        raise ValueError("Archived decisions contain duplicate link IDs.")

    candidates = sorted({int(value) for value in candidate_link_ids})
    old_lookup = old.set_index("id_link")["review_decision"]
    missing_candidates = sorted(set(candidates).difference(old_lookup.index))
    if missing_candidates:
        raise ValueError(
            "Current review candidate membership differs from the archived review; automatic migration is unsafe. "
            f"Missing archived candidates: {missing_candidates}"
        )
    seed = pd.DataFrame({"id_link": candidates})
    seed["review_decision"] = seed["id_link"].map(old_lookup).astype("string")
    affected = {int(value) for value in affected_candidate_link_ids}
    seed.loc[seed["id_link"].isin(affected), "review_decision"] = pd.NA
    return seed


def create_direction_migration(
    variant_output_dir: str | Path,
    *,
    changed_link_ids: Iterable[int],
    source_links_path: str | Path | None = None,
    review_output_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    width_field: str = "wid_adj_wet",
    len_over_width_threshold: float = 1.5,
) -> dict[str, Any]:
    # Imported lazily to keep the migration primitives independent and testable.
    from network_variants.width_review_gui import load_review_context, resolve_variant_review_paths

    paths = resolve_variant_review_paths(
        variant_output_dir,
        source_links_path=source_links_path,
        review_output_dir=review_output_dir,
    )
    archive_manifest_path = paths.review_output_dir / ARCHIVE_MANIFEST_NAME
    old_decisions_path = paths.review_output_dir / ARCHIVE_DECISIONS_NAME
    old_snapshot_path = paths.review_output_dir / ARCHIVE_LINKS_SNAPSHOT_NAME
    for label, file_path in {
        "archive manifest": archive_manifest_path,
        "archived decisions": old_decisions_path,
        "archived links snapshot": old_snapshot_path,
        "current source links": paths.source_links_path,
    }.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Required {label} does not exist: {file_path}")

    archive_manifest = json.loads(archive_manifest_path.read_text(encoding="utf-8"))
    archive_hashes = archive_manifest.get("archive_file_sha256", {}) or {}
    expected_decisions_hash = archive_hashes.get("decisions")
    expected_snapshot_hash = archive_hashes.get("links_snapshot")
    if expected_decisions_hash != _sha256(old_decisions_path):
        raise ValueError("Archived decision CSV no longer matches its recorded checksum.")
    if expected_snapshot_hash != _sha256(old_snapshot_path):
        raise ValueError("Archived links snapshot no longer matches its recorded checksum.")

    candidate_frame, _, _, _, _ = load_review_context(
        paths,
        width_field=width_field,
        len_over_width_threshold=len_over_width_threshold,
    )
    candidates = sorted(
        pd.to_numeric(
            candidate_frame.loc[candidate_frame["candidate_flag"], "id_link"], errors="coerce"
        ).dropna().astype(int).unique().tolist()
    )
    context = derive_direction_affected_context(
        gpd.read_file(old_snapshot_path),
        gpd.read_file(paths.source_links_path),
        changed_link_ids=changed_link_ids,
        candidate_link_ids=candidates,
    )
    seed = build_migrated_decision_seed(
        pd.read_csv(old_decisions_path),
        candidate_link_ids=candidates,
        affected_candidate_link_ids=context["affected_candidate_link_ids"],
    )

    migration_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (paths.variant_output_dir / DEFAULT_MIGRATION_DIR_NAME).resolve()
    )
    migration_dir.mkdir(parents=True, exist_ok=True)
    seed_path = migration_dir / DEFAULT_SEED_NAME
    manifest_path = migration_dir / DEFAULT_MANIFEST_NAME
    _atomic_write_csv(seed, seed_path)
    manifest = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "example_id": paths.example_id,
        "variant_id": paths.variant_id,
        "variant_output_dir": str(paths.variant_output_dir),
        "width_field": width_field,
        "len_over_width_threshold": float(len_over_width_threshold),
        "sources": {
            "archive_manifest": str(archive_manifest_path),
            "archive_manifest_sha256": _sha256(archive_manifest_path),
            "archived_decisions": str(old_decisions_path),
            "archived_decisions_sha256": _sha256(old_decisions_path),
            "archived_links_snapshot": str(old_snapshot_path),
            "archived_links_snapshot_sha256": _sha256(old_snapshot_path),
            "current_source_links": str(paths.source_links_path),
            "current_source_links_sha256": _sha256(paths.source_links_path),
            "variant_manifest": str(paths.manifest_path),
            "variant_manifest_sha256": _sha256(paths.manifest_path),
        },
        "impact": context,
        "candidate_count": int(len(seed)),
        "carried_decision_count": int(seed["review_decision"].notna().sum()),
        "undecided_candidate_count": int(seed["review_decision"].isna().sum()),
        "seed": {"path": str(seed_path), "sha256": _sha256(seed_path)},
        "policy": (
            "Only good/bad classifications are migrated. Direction-affected candidates are cleared. "
            "Numeric width corrections and old junction metrics are never migrated."
        ),
    }
    _atomic_write_json(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256(manifest_path)
    return manifest


def validate_direction_migration(
    manifest_path: str | Path,
    *,
    variant_output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MIGRATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported direction-migration schema in {path}.")
    current_variant = Path(variant_output_dir).expanduser().resolve()
    if Path(manifest["variant_output_dir"]).resolve() != current_variant:
        raise ValueError("Direction-migration manifest belongs to a different variant output.")
    sources = manifest.get("sources", {}) or {}
    for path_key, hash_key in (
        ("archive_manifest", "archive_manifest_sha256"),
        ("archived_decisions", "archived_decisions_sha256"),
        ("archived_links_snapshot", "archived_links_snapshot_sha256"),
        ("current_source_links", "current_source_links_sha256"),
        ("variant_manifest", "variant_manifest_sha256"),
    ):
        source = Path(sources[path_key]).expanduser().resolve()
        if not source.exists() or _sha256(source) != sources[hash_key]:
            raise ValueError(f"Direction-migration source changed after migration: {source}")
    seed_path = Path(manifest["seed"]["path"]).expanduser().resolve()
    if not seed_path.exists() or _sha256(seed_path) != manifest["seed"]["sha256"]:
        raise ValueError("Direction-migration seed is missing or has changed.")
    seed = pd.read_csv(seed_path)
    if len(seed) != int(manifest["candidate_count"]):
        raise ValueError("Direction-migration seed row count does not match its manifest.")
    return seed, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely migrate an archived width review after direction-only link reversals."
    )
    parser.add_argument("variant_output_dir")
    parser.add_argument("--changed-link-id", type=int, action="append", required=True)
    parser.add_argument("--source-links", default=None)
    parser.add_argument("--review-output-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--width-field", default="wid_adj_wet")
    parser.add_argument("--len-over-width-threshold", type=float, default=1.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = create_direction_migration(
        args.variant_output_dir,
        changed_link_ids=args.changed_link_id,
        source_links_path=args.source_links,
        review_output_dir=args.review_output_dir,
        output_dir=args.output_dir,
        width_field=args.width_field,
        len_over_width_threshold=args.len_over_width_threshold,
    )
    print(f"Wrote migration manifest to {result['manifest_path']}")
    print(
        f"Carried {result['carried_decision_count']} decisions; "
        f"reopened {result['undecided_candidate_count']} candidates: "
        f"{result['impact']['affected_candidate_link_ids']}"
    )


if __name__ == "__main__":
    main()
