from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from network_variants.island_analysis_selection import (
    build_strict_island_analysis_manifest,
)


def test_strict_manifest_retains_raw_inventory_and_records_reviewed_exclusion(tmp_path: Path) -> None:
    root = tmp_path / "15_island__anchored"
    root.mkdir()
    (root / "experiment_manifest.json").write_text(
        json.dumps({"example_id": "15"}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"island_id": "island_15_030", "analysis_backend": "island_local_patch"},
            {"island_id": "island_15_031", "analysis_backend": "island_full_raster"},
        ]
    ).to_csv(root / "island_analysis_manifest.csv", index=False)
    review = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {
                "example_id": "15",
                "island_id": "island_15_031",
                "strict_analysis_status": "exclude_pruning_or_mask_topology_exception",
                "review_code": "rivgraph_self_loop_pruning",
                "review_comment": "Raw skeleton is one-cycle; pruning removes a self-loop.",
            }
        ]
    ).to_csv(review, index=False)

    result = build_strict_island_analysis_manifest(root, review_registry_path=review)
    statuses = result.set_index("island_id")["strict_analysis_status"].to_dict()
    assert statuses == {
        "island_15_030": "include_primary_one_cycle_analysis",
        "island_15_031": "exclude_pruning_or_mask_topology_exception",
    }
    assert (root / "island_analysis_manifest.csv").exists()
    assert (root / "island_analysis_manifest_strict.csv").exists()
    summary = json.loads((root / "island_analysis_manifest_strict.json").read_text())
    assert summary["n_included"] == 1
    assert summary["n_excluded"] == 1


def test_strict_refresh_preserves_published_duplicate_exclusion(tmp_path):
    root=tmp_path/'41_island__anchored';root.mkdir()
    (root/'experiment_manifest.json').write_text(json.dumps({'example_id':'41'}))
    pd.DataFrame([dict(island_id='island_41_006',separate_observation_excluded=True)]).to_csv(root/'island_analysis_manifest.csv',index=False)
    review=tmp_path/'review.csv'
    pd.DataFrame(columns=['example_id','island_id','strict_analysis_status','review_code','review_comment']).to_csv(review,index=False)
    result=build_strict_island_analysis_manifest(root,review_registry_path=review)
    assert result.strict_analysis_status.iloc[0]=='exclude_duplicate_use_independent'
    summary=json.loads((root/'island_analysis_manifest_strict.json').read_text())
    assert summary['n_excluded']==1 and summary['n_included']==0
