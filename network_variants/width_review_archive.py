from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import geopandas as gpd
import pandas as pd


ARCHIVE_SCHEMA_VERSION = "width-review-archive-v1"
ARCHIVE_DECISIONS_NAME = "base_width_review_decisions.csv"
ARCHIVE_SUMMARY_NAME = "base_width_review_apply_summary.json"
ARCHIVE_TARGETS_NAME = "reviewed_width_targets.csv"
ARCHIVE_LINKS_SNAPSHOT_NAME = "reviewed_links_snapshot.gpkg"
ARCHIVE_MANIFEST_NAME = "base_width_review_manifest.json"
SOURCE_REVIEW_FIELDS = (
    "width_reviewed_flag",
    "width_review_decision",
    "width_update_applied",
    "width_update_method",
    "width_update_source_link_id",
    "width_update_scale_factor",
    "width_update_target_context_width",
    "wid",
    "wid_adj",
    "wid_med",
    "wid_orig",
    "wid_adj_orig",
    "wid_med_orig",
)


def normalize_example_id(value: object) -> str:
    text = str(value).strip()
    return f"{int(text):02d}" if text.isdigit() else text


def review_archive_dir(source_links_path: str | Path) -> Path:
    """Return the durable run-level archive beside, not inside, rivgraph/."""
    source = Path(source_links_path).expanduser().resolve()
    if source.parent.name != "rivgraph":
        raise ValueError(
            "Expected reviewed source links below <run>/rivgraph so the permanent "
            f"archive can be resolved, found: {source}"
        )
    return source.parent.parent / "width_review"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
        newline="",
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _variant_review_inputs(variant_output_dir: Path) -> dict[str, Path | str]:
    variant = variant_output_dir.expanduser().resolve()
    manifest_path = variant / "summary" / "variant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Variant manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_paths = manifest.get("output_paths", {}) or {}
    families_path = Path(output_paths["link_width_families"]).expanduser().resolve()
    families = pd.read_csv(families_path, nrows=1)
    if families.empty:
        raise ValueError(f"Width-family file has no rows: {families_path}")
    example_id = normalize_example_id(families.iloc[0].get("example_id"))
    review_dir = variant / "width_review"
    return {
        "variant": variant,
        "manifest": manifest_path,
        "families": families_path,
        "example_id": example_id,
        "decisions": review_dir / "width_review_decisions.csv",
        "summary": review_dir / "width_review_apply_summary.json",
    }


def _build_review_targets(
    decisions: pd.DataFrame,
    source_links: gpd.GeoDataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    if "id_link" not in decisions.columns or "review_decision" not in decisions.columns:
        raise ValueError("Review decisions must contain id_link and review_decision.")
    targets = decisions.copy()
    targets["id_link"] = pd.to_numeric(targets["id_link"], errors="raise").astype(int)
    decisions_normalized = targets["review_decision"].astype("string").str.strip().str.lower()
    decisions_normalized = decisions_normalized.mask(decisions_normalized.isin(["", "skip"]), pd.NA)
    invalid_mask = decisions_normalized.notna() & ~decisions_normalized.isin(["good", "bad"])
    if invalid_mask.any():
        invalid = sorted(decisions_normalized.loc[invalid_mask].dropna().unique())
        raise ValueError(f"Review decisions must contain only good/bad or be undecided; invalid={invalid}")
    if require_complete and decisions_normalized.isna().any():
        raise ValueError("The review must be complete and contain only good/bad decisions.")
    targets["review_decision"] = decisions_normalized

    source = source_links.copy()
    source["id_link"] = pd.to_numeric(source["id_link"], errors="raise").astype(int)
    if source["id_link"].duplicated().any():
        raise ValueError("Reviewed source links contain duplicate id_link values.")
    available_fields = [field for field in SOURCE_REVIEW_FIELDS if field in source.columns]
    source_attributes = pd.DataFrame(source.drop(columns=source.geometry.name))[
        ["id_link", *available_fields]
    ]
    targets = targets.merge(
        source_attributes,
        on="id_link",
        how="left",
        validate="1:1",
        suffixes=("", "_source"),
        indicator=True,
    )
    missing = targets.loc[targets["_merge"] != "both", "id_link"].astype(int).tolist()
    if missing:
        raise ValueError(f"Reviewed candidate IDs are absent from the source links: {missing}")
    return targets.drop(columns="_merge").sort_values("id_link", kind="mergesort").reset_index(drop=True)


def _archive_paths(archive_dir: Path) -> dict[str, Path]:
    return {
        "decisions": archive_dir / ARCHIVE_DECISIONS_NAME,
        "summary": archive_dir / ARCHIVE_SUMMARY_NAME,
        "targets": archive_dir / ARCHIVE_TARGETS_NAME,
        "links_snapshot": archive_dir / ARCHIVE_LINKS_SNAPSHOT_NAME,
        "manifest": archive_dir / ARCHIVE_MANIFEST_NAME,
    }


def version_existing_review_archive(archive_dir: str | Path) -> Path | None:
    """Preserve the current archive files before a GUI save or start-over save."""
    archive = Path(archive_dir).expanduser().resolve()
    paths = _archive_paths(archive)
    existing = [path for path in paths.values() if path.exists()]
    if not existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    version_dir = archive / "versions" / timestamp
    version_dir.mkdir(parents=True, exist_ok=False)
    for source in existing:
        _atomic_copy(source, version_dir / source.name)
    return version_dir


def finalize_width_review_archive(
    *,
    archive_dir: str | Path,
    decisions_path: str | Path,
    summary_path: str | Path,
    source_links_path: str | Path,
    example_id: str,
    variant_output_dir: str | Path,
    variant_manifest_path: str | Path,
) -> dict[str, Any]:
    """Regenerate durable derived review artifacts after a GUI save."""
    archive = Path(archive_dir).expanduser().resolve()
    decisions_file = Path(decisions_path).expanduser().resolve()
    summary_file = Path(summary_path).expanduser().resolve()
    source_links_file = Path(source_links_path).expanduser().resolve()
    for label, path in {
        "decisions": decisions_file,
        "summary": summary_file,
        "source links": source_links_file,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {label} artifact does not exist: {path}")

    paths = _archive_paths(archive)
    archive.mkdir(parents=True, exist_ok=True)
    if decisions_file != paths["decisions"]:
        _atomic_copy(decisions_file, paths["decisions"])
    if summary_file != paths["summary"]:
        _atomic_copy(summary_file, paths["summary"])

    decisions = pd.read_csv(paths["decisions"])
    source_frame = gpd.read_file(source_links_file)
    targets = _build_review_targets(decisions, source_frame, require_complete=False)
    _atomic_copy(source_links_file, paths["links_snapshot"])
    _atomic_write_csv(targets, paths["targets"])

    normalized = targets["review_decision"].astype("string").str.strip().str.lower()
    counts = normalized.value_counts().to_dict()
    undecided = int(normalized.isna().sum())
    completion = "complete" if undecided == 0 else "incomplete"
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "example_id": normalize_example_id(example_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "variant_output_dir": str(Path(variant_output_dir).expanduser().resolve()),
            "variant_manifest": str(Path(variant_manifest_path).expanduser().resolve()),
            "decisions": str(paths["decisions"]),
            "apply_summary": str(paths["summary"]),
            "reviewed_links": str(source_links_file),
        },
        "files": {key: path.name for key, path in paths.items() if key != "manifest"},
        "counts": {
            "review_rows": int(len(targets)),
            "good": int(counts.get("good", 0)),
            "bad": int(counts.get("bad", 0)),
            "undecided": undecided,
            "source_links": int(len(source_frame)),
        },
        "review_completion_status": completion,
        "archive_file_sha256": {
            key: _sha256(path) for key, path in paths.items() if key != "manifest"
        },
        "review_status": "archived_unfingerprinted",
    }
    _atomic_write_json(manifest, paths["manifest"])
    return {
        "example_id": normalize_example_id(example_id),
        "archive_dir": str(archive),
        "manifest": str(paths["manifest"]),
        "review_rows": int(len(targets)),
        "good": int(counts.get("good", 0)),
        "bad": int(counts.get("bad", 0)),
        "undecided": undecided,
        "review_completion_status": completion,
    }


def archive_base_width_review(
    variant_output_dir: str | Path,
    *,
    source_links_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy one completed base review into its durable run-level archive."""
    inputs = _variant_review_inputs(Path(variant_output_dir))
    example_id = str(inputs["example_id"])
    source_links = (
        Path(source_links_path).expanduser().resolve()
        if source_links_path is not None
        else Path(__file__).resolve().parents[1]
        / "rivgraph_centerline"
        / "outputs"
        / "runs"
        / example_id
        / "rivgraph"
        / f"{example_id}_links.gpkg"
    )
    required_sources = {
        "decisions": Path(inputs["decisions"]),
        "summary": Path(inputs["summary"]),
        "source_links": source_links,
    }
    for label, path in required_sources.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {label} artifact does not exist: {path}")

    archive_dir = review_archive_dir(source_links)
    archive_paths = _archive_paths(archive_dir)
    existing = [path for path in archive_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Width-review archive already exists; use overwrite only after validating the current review: "
            + ", ".join(str(path) for path in existing)
        )

    decisions = pd.read_csv(required_sources["decisions"])
    source_frame = gpd.read_file(source_links)
    targets = _build_review_targets(decisions, source_frame)
    archive_dir.mkdir(parents=True, exist_ok=True)
    _atomic_copy(required_sources["decisions"], archive_paths["decisions"])
    _atomic_copy(required_sources["summary"], archive_paths["summary"])
    _atomic_copy(source_links, archive_paths["links_snapshot"])
    _atomic_write_csv(targets, archive_paths["targets"])

    reviewed_counts = targets["review_decision"].value_counts().to_dict()
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "example_id": example_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "variant_output_dir": str(inputs["variant"]),
            "variant_manifest": str(inputs["manifest"]),
            "decisions": str(required_sources["decisions"]),
            "apply_summary": str(required_sources["summary"]),
            "reviewed_links": str(source_links),
        },
        "files": {key: path.name for key, path in archive_paths.items() if key != "manifest"},
        "counts": {
            "review_rows": int(len(targets)),
            "good": int(reviewed_counts.get("good", 0)),
            "bad": int(reviewed_counts.get("bad", 0)),
            "source_links": int(len(source_frame)),
        },
        "archive_file_sha256": {
            key: _sha256(path)
            for key, path in archive_paths.items()
            if key != "manifest"
        },
        "review_status": "archived_unfingerprinted",
        "notes": "Mask and structural-network freshness fingerprints are added in Phase 2.",
    }
    _atomic_write_json(manifest, archive_paths["manifest"])
    return {
        "example_id": example_id,
        "archive_dir": str(archive_dir),
        "manifest": str(archive_paths["manifest"]),
        "review_rows": int(len(targets)),
        "good": int(reviewed_counts.get("good", 0)),
        "bad": int(reviewed_counts.get("bad", 0)),
    }


def recover_base_width_review_archive(
    variant_output_dir: str | Path,
    *,
    source_links_path: str | Path,
    source_nodes_path: str | Path,
    cleaned_mask_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reconstruct a lost run-level archive from a completed base variant.

    Older experiments retain their reviewed candidate labels in the completed
    base width product even when a later storage cleanup removed the separate
    ``width_review`` folder.  This recovery is deliberately conservative: it
    accepts only a complete good/bad classification already marked as having
    come from a permanent archive, and records the exact base product from
    which the archive was recovered.
    """
    variant = Path(variant_output_dir).expanduser().resolve()
    families_path = variant / "widths" / "links_with_width_families.gpkg"
    if not families_path.exists():
        raise FileNotFoundError(f"Completed base width product does not exist: {families_path}")
    source_links = Path(source_links_path).expanduser().resolve()
    source_nodes = Path(source_nodes_path).expanduser().resolve()
    cleaned_mask = Path(cleaned_mask_path).expanduser().resolve()
    for label, path in {
        "source links": source_links,
        "source nodes": source_nodes,
        "cleaned mask": cleaned_mask,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {label} does not exist: {path}")

    families = gpd.read_file(families_path)
    required = {"id_link", "manual_width_review_candidate", "manual_width_review_decision"}
    missing = sorted(required.difference(families.columns))
    if missing:
        raise ValueError(f"Base width product cannot recover review labels; missing={missing}")
    if "manual_width_review_source" in families.columns:
        sources = families.loc[
            families["manual_width_review_candidate"].fillna(False).astype(bool),
            "manual_width_review_source",
        ].astype("string").dropna().unique().tolist()
        if sources != ["permanent_archive"]:
            raise ValueError(
                "Base width product has no unambiguous permanent-review provenance: "
                f"sources={sources}"
            )
    candidates = families.loc[
        families["manual_width_review_candidate"].fillna(False).astype(bool),
        ["id_link", "manual_width_review_decision"],
    ].copy()
    candidates["id_link"] = pd.to_numeric(candidates["id_link"], errors="raise").astype(int)
    candidates["review_decision"] = (
        candidates.pop("manual_width_review_decision").astype("string").str.strip().str.lower()
    )
    if candidates.empty or candidates["id_link"].duplicated().any():
        raise ValueError("Recovered review candidates must be non-empty with unique link IDs.")
    invalid = candidates.loc[~candidates["review_decision"].isin(["good", "bad"]), "id_link"].tolist()
    if invalid:
        raise ValueError(f"Recovered review decisions are incomplete/invalid for links: {invalid}")

    archive_dir = review_archive_dir(source_links)
    paths = _archive_paths(archive_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "A width-review archive already exists; recovery refuses to replace it: "
            + ", ".join(str(path) for path in existing)
        )
    if existing and overwrite:
        version_existing_review_archive(archive_dir)

    source_frame = gpd.read_file(source_links)
    targets = _build_review_targets(candidates, source_frame, require_complete=True)
    example_id = normalize_example_id(families.iloc[0].get("example_id", source_links.stem.split("_")[0]))
    summary = {
        "schema_version": "base-width-review-recovery-v1",
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "recovery_source_variant": str(variant),
        "recovery_source_width_product": str(families_path),
        "recovery_source_width_product_sha256": _sha256(families_path),
        "decision_count": int(len(candidates)),
        "rationale": (
            "Recovered from completed base product retaining permanent-archive review provenance; "
            "numeric reviewed width targets are retained only as historical audit fields."
        ),
    }
    archive_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(candidates.sort_values("id_link", kind="mergesort"), paths["decisions"])
    _atomic_write_json(summary, paths["summary"])
    _atomic_copy(source_links, paths["links_snapshot"])
    _atomic_write_csv(targets, paths["targets"])
    counts = candidates["review_decision"].value_counts().to_dict()
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "example_id": example_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "recovered_from_variant": str(variant),
            "recovered_from_width_product": str(families_path),
            "reviewed_links": str(source_links),
            "reviewed_nodes": str(source_nodes),
            "cleaned_mask": str(cleaned_mask),
        },
        "recovery": summary,
        "files": {key: path.name for key, path in paths.items() if key != "manifest"},
        "counts": {
            "review_rows": int(len(candidates)),
            "good": int(counts.get("good", 0)),
            "bad": int(counts.get("bad", 0)),
            "source_links": int(len(source_frame)),
        },
        "review_completion_status": "complete",
        "archive_file_sha256": {
            key: _sha256(path) for key, path in paths.items() if key != "manifest"
        },
        "review_status": "recovered_unfingerprinted",
    }
    _atomic_write_json(manifest, paths["manifest"])
    # Import locally to avoid a module-level archive/fingerprint cycle.
    from network_variants.width_review_fingerprint import upgrade_review_fingerprints

    upgrade_review_fingerprints(
        archive_dir,
        mask_path=cleaned_mask,
        links_path=source_links,
        nodes_path=source_nodes,
    )
    return {
        "example_id": example_id,
        "archive_dir": str(archive_dir),
        "manifest": str(paths["manifest"]),
        "recovered_review_rows": int(len(candidates)),
        "good": int(counts.get("good", 0)),
        "bad": int(counts.get("bad", 0)),
        "source_width_product_sha256": _sha256(families_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy completed base-width reviews into durable per-example archives."
    )
    parser.add_argument(
        "variant_output_dir",
        nargs="+",
        help="One or more S000_base variant directories containing width_review/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing archive.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = [
        archive_base_width_review(path, overwrite=args.overwrite)
        for path in args.variant_output_dir
    ]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
