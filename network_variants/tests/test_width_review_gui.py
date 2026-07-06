from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from network_variants.width_review_gui import (
    QC_TRIMMED_FALLBACK_FIELD,
    apply_width_review_updates,
    build_candidate_frame,
    compute_sample_qc_frame,
    solve_continuity_replacement,
)


def test_compute_sample_qc_frame_marks_trimmed_rows_and_aggregates() -> None:
    samples = pd.DataFrame(
        {
            "id_link": [1, 1, 1, 1],
            "sample_index": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "width_total": [2.0, 2.0, 2.0, 2.0],
            "qc_opposition_cos": [-1.0, -0.8, -0.6, -0.4],
            "width_invalid": [False, False, False, False],
        }
    )

    marked, qc = compute_sample_qc_frame(samples)

    assert marked["used_in_trimmed_width"].tolist() == [False, True, False, False]
    row = qc.iloc[0]
    assert np.isclose(float(row["qc_mean_opposition_cos"]), -0.7)
    assert np.isclose(float(row[QC_TRIMMED_FALLBACK_FIELD]), -0.8)
    assert int(row["n_trimmed_qc_samples"]) == 1


def test_solve_continuity_replacement_reduces_node_deviation() -> None:
    frame = pd.DataFrame(
        {
            "id_link": [10, 11, 12, 13],
            "id_us_node": [1, 2, 3, 2],
            "id_ds_node": [2, 3, 4, 5],
            "wid_adj_wet": [2.0, 8.0, 10.0, 2.0],
            "node_dev_max": [3.0, 3.0, 3.0, 3.0],
        }
    )
    candidate_row = frame.loc[frame["id_link"] == 11].iloc[0]

    replacement = solve_continuity_replacement(candidate_row, frame, width_field="wid_adj_wet")

    assert replacement["proposed_width"] < 8.0
    assert replacement["predicted_node_dev_max_after"] < 3.0
    assert replacement["predicted_improvement"] > 0


def test_apply_width_review_updates_scales_core_fields_and_preserves_originals(tmp_path: Path) -> None:
    links_path = tmp_path / "02_links.gpkg"
    source_links = gpd.GeoDataFrame(
        [
            {
                "id_link": 11,
                "wid": 40.0,
                "wid_adj": 20.0,
                "wid_med": 16.0,
                "id_us_node": 2,
                "id_ds_node": 3,
                "geometry": LineString([(1.0, 0.0), (2.0, 0.0)]),
            },
            {
                "id_link": 10,
                "wid": 60.0,
                "wid_adj": 30.0,
                "wid_med": 24.0,
                "id_us_node": 1,
                "id_ds_node": 2,
                "geometry": LineString([(0.0, 0.0), (1.0, 0.0)]),
            },
            {
                "id_link": 12,
                "wid": 60.0,
                "wid_adj": 30.0,
                "wid_med": 24.0,
                "id_us_node": 3,
                "id_ds_node": 4,
                "geometry": LineString([(2.0, 0.0), (3.0, 0.0)]),
            },
            {
                "id_link": 13,
                "wid": 60.0,
                "wid_adj": 30.0,
                "wid_med": 24.0,
                "id_us_node": 2,
                "id_ds_node": 5,
                "geometry": LineString([(1.0, 0.0), (1.0, -1.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    source_links.to_file(links_path, driver="GPKG")

    candidate_frame = pd.DataFrame(
        {
            "id_link": [10, 11, 12, 13],
            "id_us_node": [1, 2, 3, 2],
            "id_ds_node": [2, 3, 4, 5],
            "wid_adj_wet": [2.0, 8.0, 10.0, 2.0],
            "node_dev_max": [1.0, 3.0, 1.0, 1.0],
        }
    )
    review_decisions = pd.DataFrame(
        {
            "id_link": [11, 10],
            "review_decision": ["bad", "good"],
        }
    )

    summary = apply_width_review_updates(
        review_decisions=review_decisions,
        candidate_frame=candidate_frame,
        width_field="wid_adj_wet",
        source_links_path=links_path,
        parent_link_mapping={10: 10, 11: 11, 12: 12, 13: 13},
        overwrite_backup=False,
    )

    assert Path(summary["source_links_backup_path"]).exists()
    updated = gpd.read_file(links_path).set_index("id_link", drop=False)
    row_bad = updated.loc[11]
    assert bool(row_bad["width_reviewed_flag"]) is True
    assert row_bad["width_review_decision"] == "bad"
    assert bool(row_bad["width_update_applied"]) is True
    assert np.isclose(float(row_bad["wid_orig"]), 40.0)
    assert np.isclose(float(row_bad["wid_adj_orig"]), 20.0)
    assert np.isclose(float(row_bad["wid_med_orig"]), 16.0)
    assert float(row_bad["wid"]) < 40.0
    assert float(row_bad["wid_adj"]) < 20.0
    assert float(row_bad["wid_med"]) < 16.0

    row_good = updated.loc[10]
    assert bool(row_good["width_reviewed_flag"]) is True
    assert row_good["width_review_decision"] == "good"
    assert bool(row_good["width_update_applied"]) is False
