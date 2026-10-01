from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd

from network_variants.width_review_archive import (
    ARCHIVE_DECISIONS_NAME,
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_TARGETS_NAME,
    _sha256,
    review_archive_dir,
)
from network_variants.width_review_fingerprint import (
    CORRUPT_STATUS,
    CURRENT_STATUS,
    DIRECTION_ONLY_CURRENT_STATUS,
    validate_direction_only_review_continuity,
    validate_review_fingerprints,
)


BASE_REVIEW_APPLICATION_VERSION = "base-width-review-application-v1"
BASE_REVIEW_PROVENANCE_VERSION = "base-width-review-provenance-v1"

# These are summary widths derived from the same cross-section samples.  The
# manual review scale is defined against wid_adj_wet and therefore applies to
# their dimensional width statistics, but not to fractions, counts, or QC.
REVIEW_SCALABLE_FAMILY_FIELDS = (
    "wid_adj_total",
    "wid_med_total",
    "wid_p05_total",
    "wid_p50_total",
    "wid_p95_total",
    "wid_adj_wet",
    "wid_med_wet",
    "wid_p05_wet",
    "wid_p50_wet",
    "wid_p95_wet",
    "wid_adj_dry",
    "wid_med_dry",
    "wid_p05_dry",
    "wid_p50_dry",
    "wid_p95_dry",
)


def apply_review_scales_to_width_families(
    enriched_links: gpd.GeoDataFrame,
    review_targets: pd.DataFrame,
    *,
    scalable_fields: Iterable[str] = REVIEW_SCALABLE_FAMILY_FIELDS,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Apply reviewed wet-width scales while retaining every raw family value."""
    if "id_link" not in enriched_links.columns:
        raise ValueError("Width-family links are missing id_link.")
    required = {
        "id_link",
        "review_decision",
        "width_update_applied",
        "width_update_scale_factor",
        "width_update_target_context_width",
    }
    missing = required.difference(review_targets.columns)
    if missing:
        raise ValueError(f"Review targets are missing required fields: {sorted(missing)}")

    result = enriched_links.copy()
    result["id_link"] = pd.to_numeric(result["id_link"], errors="coerce").astype("Int64")
    if result["id_link"].isna().any() or result["id_link"].duplicated().any():
        raise ValueError("Width-family link IDs must be complete and unique.")
    result["id_link"] = result["id_link"].astype(int)

    targets = review_targets.copy()
    targets["id_link"] = pd.to_numeric(targets["id_link"], errors="coerce").astype("Int64")
    targets = targets.dropna(subset=["id_link"]).copy()
    targets["id_link"] = targets["id_link"].astype(int)
    if targets["id_link"].duplicated().any():
        raise ValueError("Review targets contain duplicate link IDs.")
    unknown = sorted(set(targets["id_link"]).difference(result["id_link"]))
    if unknown:
        raise ValueError(f"Reviewed links are absent from the base width-family output: {unknown}")

    targets["review_decision"] = targets["review_decision"].astype("string").str.strip().str.lower()
    applied = targets["width_update_applied"].fillna(False).astype(bool)
    scales = pd.to_numeric(targets["width_update_scale_factor"], errors="coerce")
    bad = targets["review_decision"].eq("bad")
    invalid = applied & (~bad | ~np.isfinite(scales) | scales.le(0))
    if invalid.any():
        invalid_ids = targets.loc[invalid, "id_link"].astype(int).tolist()
        raise ValueError(f"Invalid applied width-review scales for links: {invalid_ids}")
    inconsistent_bad = bad & ~applied
    if inconsistent_bad.any():
        invalid_ids = targets.loc[inconsistent_bad, "id_link"].astype(int).tolist()
        raise ValueError(f"Bad review decisions lack an applied correction for links: {invalid_ids}")

    active = targets.loc[applied].copy()
    active["width_update_scale_factor"] = scales.loc[applied].astype(float)
    active_lookup = active.set_index("id_link")
    result_index = result.set_index("id_link", drop=False)
    fields = [field for field in scalable_fields if field in result.columns]
    for field in fields:
        raw_field = f"{field}_raw"
        if raw_field in result.columns:
            raise ValueError(
                f"Width-family output already contains {raw_field}; refusing to apply a review twice."
            )
        result[raw_field] = pd.to_numeric(result[field], errors="coerce")

    result["family_width_review_applied"] = False
    result["family_width_review_scale_factor"] = np.nan
    result["family_width_review_target_wet_width"] = np.nan
    result["family_width_review_method"] = pd.Series(pd.NA, index=result.index, dtype="object")

    applied_records: list[dict[str, Any]] = []
    for link_id, target in active_lookup.iterrows():
        row_index = result_index.index.get_loc(int(link_id))
        # get_loc is positional because IDs are unique; convert it to the
        # original RangeIndex/label used by result.loc.
        output_index = result.index[int(row_index)]
        scale = float(target["width_update_scale_factor"])
        for field in fields:
            raw_value = pd.to_numeric(pd.Series([result.at[output_index, f"{field}_raw"]]), errors="coerce").iloc[0]
            if np.isfinite(raw_value):
                result.at[output_index, field] = float(raw_value) * scale
        wet_target = pd.to_numeric(
            pd.Series([target["width_update_target_context_width"]]), errors="coerce"
        ).iloc[0]
        corrected_wet = pd.to_numeric(pd.Series([result.at[output_index, "wid_adj_wet"]]), errors="coerce").iloc[0]
        if np.isfinite(wet_target) and np.isfinite(corrected_wet) and not np.isclose(
            float(wet_target), float(corrected_wet), rtol=1e-9, atol=1e-9
        ):
            raise ValueError(
                f"Reviewed target mismatch for link {link_id}: expected {wet_target}, calculated {corrected_wet}."
            )
        result.at[output_index, "family_width_review_applied"] = True
        result.at[output_index, "family_width_review_scale_factor"] = scale
        result.at[output_index, "family_width_review_target_wet_width"] = wet_target
        result.at[output_index, "family_width_review_method"] = target.get(
            "width_update_method", "continuity_local_endpoint_context"
        )
        applied_records.append(
            {
                "id_link": int(link_id),
                "scale_factor": scale,
                "raw_wid_adj_wet": float(result.at[output_index, "wid_adj_wet_raw"]),
                "corrected_wid_adj_wet": float(result.at[output_index, "wid_adj_wet"]),
            }
        )

    return result, {
        "application_version": BASE_REVIEW_APPLICATION_VERSION,
        "status": "applied",
        "reviewed_candidate_count": int(len(targets)),
        "applied_correction_count": int(len(active)),
        "applied_link_ids": [record["id_link"] for record in applied_records],
        "scaled_fields": fields,
        "raw_field_suffix": "_raw",
        "records": applied_records,
    }


def apply_permanent_base_review(
    enriched_links: gpd.GeoDataFrame,
    *,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    cleaned_mask_path: str | Path,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Apply a current complete run-level review, or explicitly report why it was skipped."""
    reviewed_links = Path(reviewed_links_path).expanduser().resolve()
    archive_dir = review_archive_dir(reviewed_links)
    manifest_path = archive_dir / ARCHIVE_MANIFEST_NAME
    if not manifest_path.exists():
        return enriched_links, {
            "application_version": BASE_REVIEW_APPLICATION_VERSION,
            "status": "not_available",
            "archive_dir": str(archive_dir),
            "applied_correction_count": 0,
        }

    validation = validate_review_fingerprints(
        archive_dir,
        mask_path=cleaned_mask_path,
        links_path=reviewed_links,
        nodes_path=reviewed_nodes_path,
    )
    status = str(validation.get("status"))
    if status == CORRUPT_STATUS:
        raise ValueError(
            f"Permanent width-review archive is corrupt and cannot be used: {archive_dir}; "
            f"issues={validation.get('issues', [])}"
        )
    if status != CURRENT_STATUS:
        return enriched_links, {
            "application_version": BASE_REVIEW_APPLICATION_VERSION,
            "status": "skipped_stale_or_incomplete",
            "review_status": status,
            "archive_dir": str(archive_dir),
            "issues": validation.get("issues", []),
            "applied_correction_count": 0,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("review_completion_status") != "complete":
        return enriched_links, {
            "application_version": BASE_REVIEW_APPLICATION_VERSION,
            "status": "skipped_stale_or_incomplete",
            "review_status": "incomplete_review",
            "archive_dir": str(archive_dir),
            "issues": ["Permanent review is not complete."],
            "applied_correction_count": 0,
        }
    targets_name = (manifest.get("files", {}) or {}).get("targets", ARCHIVE_TARGETS_NAME)
    targets_path = archive_dir / targets_name
    adjusted, summary = apply_review_scales_to_width_families(
        enriched_links,
        pd.read_csv(targets_path),
    )
    summary.update(
        {
            "archive_dir": str(archive_dir),
            "archive_manifest": str(manifest_path),
            "archive_manifest_sha256": _sha256(manifest_path),
            "review_targets": str(targets_path),
            "review_targets_sha256": _sha256(targets_path),
            "review_status": status,
        }
    )
    return adjusted, summary


def attach_permanent_base_review_provenance(
    enriched_links: gpd.GeoDataFrame,
    *,
    reviewed_links_path: str | Path,
    reviewed_nodes_path: str | Path,
    cleaned_mask_path: str | Path,
    direction_review_continuity_path: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Attach durable human labels without applying historical numeric targets."""
    if "id_link" not in enriched_links.columns:
        raise ValueError("Width-family links are missing id_link.")
    result = enriched_links.copy()
    result["id_link"] = pd.to_numeric(result["id_link"], errors="coerce").astype("Int64")
    if result["id_link"].isna().any() or result["id_link"].duplicated().any():
        raise ValueError("Width-family link IDs must be complete and unique.")
    result["id_link"] = result["id_link"].astype(int)
    result["manual_width_review_candidate"] = False
    result["manual_width_review_decision"] = pd.Series(pd.NA, index=result.index, dtype="object")
    result["manual_width_review_requires_correction"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    result["manual_width_review_source"] = pd.Series(pd.NA, index=result.index, dtype="object")

    reviewed_links = Path(reviewed_links_path).expanduser().resolve()
    archive_dir = review_archive_dir(reviewed_links)
    manifest_path = archive_dir / ARCHIVE_MANIFEST_NAME
    # A direction-continuity record is opt-in and lives beside the immutable
    # review archive. Ordinary runs have no such file and therefore retain the
    # normal strict freshness behaviour.
    resolved_continuity_path = (
        Path(direction_review_continuity_path).expanduser().resolve()
        if direction_review_continuity_path is not None
        else archive_dir / "direction_only_continuity.json"
    )
    common_summary = {
        "application_version": BASE_REVIEW_PROVENANCE_VERSION,
        "archive_dir": str(archive_dir),
        "numeric_targets_applied": False,
        "measured_width_fields_modified": False,
        "attached_fields": [
            "manual_width_review_candidate",
            "manual_width_review_decision",
            "manual_width_review_requires_correction",
            "manual_width_review_source",
        ],
    }
    if not manifest_path.exists():
        return result, {
            **common_summary,
            "status": "not_available",
            "reviewed_candidate_count": 0,
        }

    validation = validate_review_fingerprints(
        archive_dir,
        mask_path=cleaned_mask_path,
        links_path=reviewed_links,
        nodes_path=reviewed_nodes_path,
    )
    if resolved_continuity_path.exists() and validation.get("status") != CURRENT_STATUS:
        validation = validate_direction_only_review_continuity(
            archive_dir,
            continuity_path=resolved_continuity_path,
            mask_path=cleaned_mask_path,
            links_path=reviewed_links,
            nodes_path=reviewed_nodes_path,
        )
    status = str(validation.get("status"))
    if status == CORRUPT_STATUS:
        raise ValueError(
            f"Permanent width-review archive is corrupt and cannot be used: {archive_dir}; "
            f"issues={validation.get('issues', [])}"
        )
    if status not in {CURRENT_STATUS, DIRECTION_ONLY_CURRENT_STATUS}:
        return result, {
            **common_summary,
            "status": "skipped_stale_or_incomplete",
            "review_status": status,
            "issues": validation.get("issues", []),
            "reviewed_candidate_count": 0,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Archives created before review_completion_status was introduced were
    # complete by construction. This matches the fingerprint validator's
    # backwards-compatible interpretation.
    completion_status = str(manifest.get("review_completion_status", "complete"))
    if completion_status != "complete":
        return result, {
            **common_summary,
            "status": "skipped_stale_or_incomplete",
            "review_status": "incomplete_review",
            "issues": ["Permanent review is not complete."],
            "reviewed_candidate_count": 0,
        }

    decisions_name = (manifest.get("files", {}) or {}).get("decisions", ARCHIVE_DECISIONS_NAME)
    decisions_path = archive_dir / decisions_name
    decisions = pd.read_csv(decisions_path)
    required = {"id_link", "review_decision"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Archived review decisions are missing required fields: {missing}")
    decisions["id_link"] = pd.to_numeric(decisions["id_link"], errors="coerce").astype("Int64")
    if decisions["id_link"].isna().any() or decisions["id_link"].duplicated().any():
        raise ValueError("Archived review decision link IDs must be complete and unique.")
    decisions["id_link"] = decisions["id_link"].astype(int)
    decisions["review_decision"] = decisions["review_decision"].astype("string").str.strip().str.lower()
    invalid_decisions = sorted(
        decisions.loc[~decisions["review_decision"].isin(["good", "bad"]), "review_decision"]
        .dropna()
        .unique()
    )
    if decisions["review_decision"].isna().any() or invalid_decisions:
        raise ValueError(
            "Archived permanent review is not a complete good/bad classification; "
            f"invalid={invalid_decisions}."
        )
    unknown = sorted(set(decisions["id_link"]).difference(result["id_link"]))
    if unknown:
        raise ValueError(f"Archived reviewed links are absent from the base output: {unknown}")

    lookup = decisions.set_index("id_link")["review_decision"]
    reviewed_mask = result["id_link"].isin(lookup.index)
    result.loc[reviewed_mask, "manual_width_review_candidate"] = True
    result.loc[reviewed_mask, "manual_width_review_decision"] = (
        result.loc[reviewed_mask, "id_link"].map(lookup).astype("object")
    )
    correction = result.loc[reviewed_mask, "manual_width_review_decision"].eq("bad").astype(int)
    result.loc[reviewed_mask, "manual_width_review_requires_correction"] = correction.to_numpy()
    result.loc[reviewed_mask, "manual_width_review_source"] = "permanent_archive"

    counts = decisions["review_decision"].value_counts().to_dict()
    return result, {
        **common_summary,
        "status": (
            "attached_direction_only_continuity"
            if status == DIRECTION_ONLY_CURRENT_STATUS
            else "attached"
        ),
        "review_status": status,
        "review_completion_status": completion_status,
        "legacy_completion_status_assumed": "review_completion_status" not in manifest,
        "archive_manifest": str(manifest_path),
        "archive_manifest_sha256": _sha256(manifest_path),
        "review_decisions": str(decisions_path),
        "review_decisions_sha256": _sha256(decisions_path),
        "reviewed_candidate_count": int(len(decisions)),
        "good_count": int(counts.get("good", 0)),
        "bad_count": int(counts.get("bad", 0)),
        "direction_review_continuity_path": (
            str(resolved_continuity_path)
            if resolved_continuity_path.exists()
            else None
        ),
        "direction_continuity_audit": validation.get("direction_continuity_audit"),
    }
