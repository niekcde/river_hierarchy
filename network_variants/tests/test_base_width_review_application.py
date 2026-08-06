from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from network_variants.base_width_review_application import (
    REVIEW_SCALABLE_FAMILY_FIELDS,
    attach_permanent_base_review_provenance,
    apply_review_scales_to_width_families,
)
import network_variants.base_width_review_application as application_module


def _families() -> gpd.GeoDataFrame:
    records: dict[str, object] = {
        "id_link": [10, 11, 12],
        "wid": [50.0, 60.0, 70.0],
    }
    for field in REVIEW_SCALABLE_FAMILY_FIELDS:
        records[field] = [100.0, 200.0, 300.0]
    return gpd.GeoDataFrame(
        records,
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
        crs="EPSG:28992",
    )


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_link": [10, 11],
            "review_decision": ["good", "bad"],
            "width_update_applied": [False, True],
            "width_update_scale_factor": [np.nan, 0.5],
            "width_update_target_context_width": [np.nan, 100.0],
            "width_update_method": [pd.NA, "continuity_local_endpoint_context"],
        }
    )


def test_review_scale_corrects_family_summaries_and_preserves_raw_values() -> None:
    result, summary = apply_review_scales_to_width_families(_families(), _targets())
    indexed = result.set_index("id_link")

    assert indexed.at[11, "wid_adj_wet_raw"] == 200.0
    assert indexed.at[11, "wid_adj_wet"] == 100.0
    assert indexed.at[11, "wid_p95_total"] == 100.0
    assert indexed.at[10, "wid_adj_wet"] == 100.0
    assert indexed.at[12, "wid_adj_wet"] == 300.0
    assert indexed.at[11, "wid"] == 60.0  # legacy fields were already corrected upstream
    assert bool(indexed.at[11, "family_width_review_applied"]) is True
    assert bool(indexed.at[10, "family_width_review_applied"]) is False
    assert summary["applied_correction_count"] == 1
    assert summary["applied_link_ids"] == [11]


def test_review_scale_rejects_bad_decision_without_correction() -> None:
    targets = _targets()
    targets.loc[targets["id_link"] == 11, "width_update_applied"] = False
    with pytest.raises(ValueError, match="lack an applied correction"):
        apply_review_scales_to_width_families(_families(), targets)


def test_review_scale_rejects_second_application() -> None:
    first, _ = apply_review_scales_to_width_families(_families(), _targets())
    with pytest.raises(ValueError, match="refusing to apply a review twice"):
        apply_review_scales_to_width_families(first, _targets())


def test_review_scale_rejects_unknown_reviewed_link() -> None:
    targets = _targets()
    targets.loc[len(targets)] = [99, "bad", True, 0.5, 25.0, "continuity"]
    with pytest.raises(ValueError, match="absent from the base"):
        apply_review_scales_to_width_families(_families(), targets)


def test_provenance_attachment_accepts_legacy_complete_archive_without_scaling(
    monkeypatch,
    tmp_path,
) -> None:
    run_dir = tmp_path / "runs" / "99"
    rivgraph_dir = run_dir / "rivgraph"
    archive_dir = run_dir / "width_review"
    rivgraph_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)
    reviewed_links = rivgraph_dir / "99_links.gpkg"
    reviewed_nodes = rivgraph_dir / "99_nodes.gpkg"
    cleaned_mask = run_dir / "99_cleaned.tif"
    for path in (reviewed_links, reviewed_nodes, cleaned_mask):
        path.touch()
    decisions_path = archive_dir / "base_width_review_decisions.csv"
    pd.DataFrame(
        {"id_link": [10, 11], "review_decision": ["good", "bad"]}
    ).to_csv(decisions_path, index=False)
    # Legacy archives did not yet include review_completion_status.
    (archive_dir / "base_width_review_manifest.json").write_text(
        '{"files": {"decisions": "base_width_review_decisions.csv"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        application_module,
        "validate_review_fingerprints",
        lambda *args, **kwargs: {"status": "current", "issues": []},
    )
    source = _families()
    measured_before = source[list(REVIEW_SCALABLE_FAMILY_FIELDS)].copy()

    result, summary = attach_permanent_base_review_provenance(
        source,
        reviewed_links_path=reviewed_links,
        reviewed_nodes_path=reviewed_nodes,
        cleaned_mask_path=cleaned_mask,
    )
    indexed = result.set_index("id_link")

    pd.testing.assert_frame_equal(
        result[list(REVIEW_SCALABLE_FAMILY_FIELDS)], measured_before
    )
    assert indexed.at[10, "manual_width_review_decision"] == "good"
    assert indexed.at[11, "manual_width_review_decision"] == "bad"
    assert int(indexed.at[11, "manual_width_review_requires_correction"]) == 1
    assert not bool(indexed.at[12, "manual_width_review_candidate"])
    assert summary["status"] == "attached"
    assert summary["legacy_completion_status_assumed"] is True
    assert summary["numeric_targets_applied"] is False
    assert summary["measured_width_fields_modified"] is False
