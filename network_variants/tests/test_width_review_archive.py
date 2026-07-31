from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString

from network_variants.width_review_archive import (
    ARCHIVE_DECISIONS_NAME,
    ARCHIVE_LINKS_SNAPSHOT_NAME,
    ARCHIVE_MANIFEST_NAME,
    ARCHIVE_SUMMARY_NAME,
    ARCHIVE_TARGETS_NAME,
    archive_base_width_review,
    finalize_width_review_archive,
    review_archive_dir,
    version_existing_review_archive,
)


def _write_review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_root = tmp_path / "rivgraph_centerline" / "outputs" / "runs" / "02"
    source_links_path = run_root / "rivgraph" / "02_links.gpkg"
    source_links_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {
            "id_link": [1, 2],
            "id_us_node": [10, 11],
            "id_ds_node": [11, 12],
            "wid": [5.0, 7.0],
            "wid_adj": [5.0, 8.0],
            "wid_med": [5.0, 8.0],
            "width_reviewed_flag": [True, True],
            "width_review_decision": ["good", "bad"],
            "width_update_applied": [False, True],
            "width_update_scale_factor": [float("nan"), 2.0],
            "width_update_target_context_width": [float("nan"), 8.0],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:32611",
    ).to_file(source_links_path, driver="GPKG")

    variant = tmp_path / "network_variants" / "output" / "02" / "states" / "S000_base" / "variant"
    (variant / "summary").mkdir(parents=True)
    (variant / "widths").mkdir(parents=True)
    (variant / "width_review").mkdir(parents=True)
    families_path = variant / "widths" / "link_width_families.csv"
    pd.DataFrame(
        {
            "example_id": ["02", "02"],
            "variant_id": ["S000_base", "S000_base"],
            "id_link": [1, 2],
        }
    ).to_csv(families_path, index=False)
    (variant / "summary" / "variant_manifest.json").write_text(
        json.dumps({"output_paths": {"link_width_families": str(families_path)}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "id_link": [1, 2],
            "len_over_width": [2.0, 0.8],
            "split_sensitive": [False, True],
            "review_decision": ["good", "bad"],
        }
    ).to_csv(variant / "width_review" / "width_review_decisions.csv", index=False)
    (variant / "width_review" / "width_review_apply_summary.json").write_text(
        json.dumps({"example_id": "02", "n_applied_updates": 1}),
        encoding="utf-8",
    )
    return variant, source_links_path


def test_review_archive_dir_is_run_level(tmp_path: Path) -> None:
    source = tmp_path / "runs" / "02" / "rivgraph" / "02_links.gpkg"
    assert review_archive_dir(source) == (tmp_path / "runs" / "02" / "width_review").resolve()


def test_archive_base_width_review_copies_and_verifies_complete_review(tmp_path: Path) -> None:
    variant, source_links = _write_review_fixture(tmp_path)

    result = archive_base_width_review(variant, source_links_path=source_links)

    archive = Path(result["archive_dir"])
    assert result["review_rows"] == 2
    assert result["good"] == 1
    assert result["bad"] == 1
    for name in (
        ARCHIVE_DECISIONS_NAME,
        ARCHIVE_SUMMARY_NAME,
        ARCHIVE_TARGETS_NAME,
        ARCHIVE_LINKS_SNAPSHOT_NAME,
        ARCHIVE_MANIFEST_NAME,
    ):
        assert (archive / name).exists()
    targets = pd.read_csv(archive / ARCHIVE_TARGETS_NAME)
    bad = targets.loc[targets["id_link"] == 2].iloc[0]
    assert bad["review_decision"] == "bad"
    assert bad["width_update_target_context_width"] == pytest.approx(8.0)
    manifest = json.loads((archive / ARCHIVE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["counts"] == {"review_rows": 2, "good": 1, "bad": 1, "source_links": 2}
    assert set(manifest["archive_file_sha256"]) == {
        "decisions",
        "summary",
        "targets",
        "links_snapshot",
    }
    assert (variant / "width_review" / "width_review_decisions.csv").exists()
    assert source_links.exists()


def test_archive_refuses_to_replace_existing_files_without_overwrite(tmp_path: Path) -> None:
    variant, source_links = _write_review_fixture(tmp_path)
    archive_base_width_review(variant, source_links_path=source_links)

    with pytest.raises(FileExistsError, match="already exists"):
        archive_base_width_review(variant, source_links_path=source_links)


def test_archive_rejects_incomplete_review(tmp_path: Path) -> None:
    variant, source_links = _write_review_fixture(tmp_path)
    decisions_path = variant / "width_review" / "width_review_decisions.csv"
    decisions = pd.read_csv(decisions_path)
    decisions.loc[0, "review_decision"] = pd.NA
    decisions.to_csv(decisions_path, index=False)

    with pytest.raises(ValueError, match="complete"):
        archive_base_width_review(variant, source_links_path=source_links)


def test_gui_finalizer_records_resumable_incomplete_review(tmp_path: Path) -> None:
    variant, source_links = _write_review_fixture(tmp_path)
    archive = review_archive_dir(source_links)
    archive.mkdir(parents=True)
    decisions = pd.read_csv(variant / "width_review" / "width_review_decisions.csv")
    decisions.loc[0, "review_decision"] = pd.NA
    decisions.to_csv(archive / ARCHIVE_DECISIONS_NAME, index=False)
    summary = archive / ARCHIVE_SUMMARY_NAME
    summary.write_text("{}", encoding="utf-8")

    result = finalize_width_review_archive(
        archive_dir=archive,
        decisions_path=archive / ARCHIVE_DECISIONS_NAME,
        summary_path=summary,
        source_links_path=source_links,
        example_id="02",
        variant_output_dir=variant,
        variant_manifest_path=variant / "summary" / "variant_manifest.json",
    )

    assert result["review_completion_status"] == "incomplete"
    assert result["undecided"] == 1
    manifest = json.loads((archive / ARCHIVE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["counts"]["undecided"] == 1
    assert manifest["review_completion_status"] == "incomplete"


def test_version_existing_archive_preserves_previous_files(tmp_path: Path) -> None:
    variant, source_links = _write_review_fixture(tmp_path)
    result = archive_base_width_review(variant, source_links_path=source_links)
    archive = Path(result["archive_dir"])
    previous_decisions = (archive / ARCHIVE_DECISIONS_NAME).read_bytes()

    version_dir = version_existing_review_archive(archive)
    assert version_dir is not None
    assert (version_dir / ARCHIVE_DECISIONS_NAME).read_bytes() == previous_decisions
    assert (version_dir / ARCHIVE_MANIFEST_NAME).exists()
