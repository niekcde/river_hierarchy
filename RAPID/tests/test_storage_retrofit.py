from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from rapid_tools.engine import write_qout_netcdf
from rapid_tools.storage_retrofit import (
    RETROFIT_ACTION,
    build_rapid_storage_retrofit_manifest,
    execute_rapid_storage_retrofit,
)


def _write_retrofit_state(
    root: Path,
    *,
    status: str = "ran",
    experiment_name: str = "56_indep__anchored",
) -> Path:
    experiment = root / experiment_name
    prep = experiment / "states" / "S001_unit_1" / "rapid" / "prep"
    run = prep.parent / "run"
    prep.mkdir(parents=True)
    run.mkdir()
    gpd.GeoDataFrame(
        {"reach_id": [1]}, geometry=[LineString([(0, 0), (1, 0)])], crs="EPSG:3857"
    ).to_file(prep / "rapid_reach_attributes.gpkg", driver="GPKG")
    pd.DataFrame({"reach_id": [1], "is_outlet": [True]}).to_csv(
        prep / "rapid_link_attributes.csv", index=False
    )
    (prep / "regenerable_debug_payload.bin").write_bytes(b"x" * 10_000)
    qout = run / "Qout_rapid_framework.nc"
    write_qout_netcdf(qout, np.array([1]), np.array([0.0, 300.0]), np.array([[1.0], [2.0]]))
    hydro = run / "outlet_hydrograph.csv"
    metrics_csv = run / "hydrograph_metrics.csv"
    metrics_json = run / "hydrograph_metrics.json"
    hydro.write_text("time_seconds,q_outlet_cms\n0,1\n300,2\n")
    metrics_csv.write_text("peak_discharge_cms\n2\n")
    metrics_json.write_text(json.dumps({"peak_discharge_cms": 2}))
    pd.DataFrame(
        [{
            "state_id": "S001_unit_1",
            "rapid_prep_dir": str(prep),
            "rapid_run_dir": str(run),
            "qout_nc": str(qout),
            "status": status,
            "hydrograph_status": "computed",
            "outlet_hydrograph_csv": str(hydro),
            "hydrograph_metrics_csv": str(metrics_csv),
            "hydrograph_metrics_json": str(metrics_json),
        }]
    ).to_csv(experiment / "rapid_run_registry.csv", index=False)
    return experiment


def test_retrofit_manifest_identifies_only_complete_routed_states(tmp_path: Path) -> None:
    _write_retrofit_state(tmp_path)
    _write_retrofit_state(
        tmp_path,
        status="failed",
        experiment_name="56_compound_bubbles__anchored",
    )

    result = build_rapid_storage_retrofit_manifest(tmp_path)

    assert int(result.summary["n_eligible"]) == 1
    assert result.rows.loc[result.rows["eligibility"].eq("eligible"), "recommended_action"].iloc[0] == RETROFIT_ACTION
    assert int(result.summary["raw_qout_bytes_eligible"]) > 0


def test_retrofit_executor_dry_run_preserves_existing_state(tmp_path: Path) -> None:
    experiment = _write_retrofit_state(tmp_path)
    manifest = build_rapid_storage_retrofit_manifest(tmp_path)
    manifest_path = tmp_path / "rapid_storage_retrofit_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_rapid_storage_retrofit(manifest_path)

    assert result.rows.loc[0, "outcome"] == "planned"
    assert (experiment / "states" / "S001_unit_1" / "rapid" / "prep").exists()
    assert (experiment / "states" / "S001_unit_1" / "rapid" / "run" / "Qout_rapid_framework.nc").exists()


def test_retrofit_executor_compacts_only_after_preserving_analysis_inputs(tmp_path: Path) -> None:
    experiment = _write_retrofit_state(tmp_path)
    manifest = build_rapid_storage_retrofit_manifest(tmp_path)
    manifest_path = tmp_path / "rapid_storage_retrofit_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_rapid_storage_retrofit(manifest_path, execute=True)

    run = experiment / "states" / "S001_unit_1" / "rapid" / "run"
    assert result.rows.loc[0, "outcome"] == "compacted"
    assert (run / "Qout_rapid_framework.nc.gz").exists()
    assert (run / "rapid_reach_attributes.gpkg").exists()
    assert (run / "rapid_link_mapping.parquet").exists()
    assert not (run / "Qout_rapid_framework.nc").exists()
    assert not (experiment / "states" / "S001_unit_1" / "rapid" / "prep").exists()
    assert int(result.rows.loc[0, "bytes_reclaimed"]) > 0


def test_retrofit_executor_normalizes_mixed_lineage_metadata_for_parquet(tmp_path: Path) -> None:
    experiment = _write_retrofit_state(tmp_path)
    mapping_path = experiment / "states" / "S001_unit_1" / "rapid" / "prep" / "rapid_link_attributes.csv"
    pd.DataFrame(
        {
            "reach_id": [1, 2],
            "compound_unit_ids": [float("nan"), "51, 52"],
            "matched_parent_link_ids": [121, "121, 122"],
        }
    ).to_csv(mapping_path, index=False)
    manifest = build_rapid_storage_retrofit_manifest(tmp_path)
    manifest_path = tmp_path / "rapid_storage_retrofit_manifest.csv"
    manifest.rows.to_csv(manifest_path, index=False)

    result = execute_rapid_storage_retrofit(manifest_path, execute=True)

    mapping = pd.read_parquet(
        experiment / "states" / "S001_unit_1" / "rapid" / "run" / "rapid_link_mapping.parquet"
    )
    assert result.rows.loc[0, "outcome"] == "compacted"
    assert pd.isna(mapping.loc[0, "compound_unit_ids"])
    assert mapping.loc[1, "compound_unit_ids"] == "51, 52"
    assert mapping["matched_parent_link_ids"].tolist() == ["121", "121, 122"]
