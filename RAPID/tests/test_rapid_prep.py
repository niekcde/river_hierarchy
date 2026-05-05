from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.io import netcdf_file
from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RAPID" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rapid_tools.engine import run_prepared_experiment
from rapid_tools.prep import RapidPrepConfig, prepare_experiment


def _write_state(tmp_path: Path, *, link_id: int = 11) -> Path:
    experiment_dir = tmp_path / "experiment"
    state_dir = experiment_dir / "states" / "S001_unit_1"
    variant_dir = state_dir / "variant"
    directed_dir = variant_dir / "directed"
    widths_dir = variant_dir / "widths"
    directed_dir.mkdir(parents=True)
    widths_dir.mkdir(parents=True)

    nodes = gpd.GeoDataFrame(
        {
            "id_node": [1, 2],
            "is_inlet": [True, False],
            "is_outlet": [False, True],
            "sword_wse": [10.0, 9.0],
            "sword_wse_field": ["wse_obs_p50", "wse_obs_p50"],
            "sword_wse_fill_method": ["requested_field", "requested_field"],
            "sword_wse_fallback_used": [False, False],
            "sword_node_id": [101, 102],
        },
        geometry=[Point(0.0, 0.0), Point(1000.0, 0.0)],
        crs="EPSG:3857",
    )
    links = gpd.GeoDataFrame(
        {
            "id_link": [link_id],
            "id_us_node": [1],
            "id_ds_node": [2],
            "is_inlet": [True],
            "is_outlet": [True],
            "len": [1000.0],
            "wid_adj_wet": [50.0],
            "wid_adj_total": [60.0],
        },
        geometry=[LineString([(0.0, 0.0), (1000.0, 0.0)])],
        crs="EPSG:3857",
    )

    directed_links_path = directed_dir / "test_directed_links.gpkg"
    directed_nodes_path = directed_dir / "test_directed_nodes.gpkg"
    widths_path = widths_dir / "links_with_width_families.gpkg"
    links.to_file(directed_links_path, driver="GPKG")
    nodes.to_file(directed_nodes_path, driver="GPKG")
    links.to_file(widths_path, driver="GPKG")

    registry = pd.DataFrame(
        [
            {
                "state_id": "S001_unit_1",
                "parent_state_id": "S000_base",
                "depth": 1,
                "mode": "independent-units",
                "state_role": "derived",
                "state_dir": str(state_dir.resolve()),
                "variant_output_dir": str(variant_dir.resolve()),
                "directed_links_path": str(directed_links_path.resolve()),
                "directed_nodes_path": str(directed_nodes_path.resolve()),
                "status": "complete",
            }
        ]
    )
    registry.to_csv(experiment_dir / "state_registry.csv", index=False)

    forcing = pd.DataFrame(
        {
            "time": [
                "2020-01-01T00:00:00Z",
                "2020-01-01T01:00:00Z",
                "2020-01-01T02:00:00Z",
            ],
            "discharge": [100.0, 110.0, 120.0],
        }
    )
    forcing.to_csv(experiment_dir / "forcing.csv", index=False)
    return experiment_dir


def test_prepare_experiment_writes_state_outputs(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    assert registry.loc[0, "status"] == "prepared"

    prep_dir = experiment_dir / "states" / "S001_unit_1" / "rapid" / "prep"
    assert (prep_dir / "rapid_prep_manifest.json").exists()
    assert (prep_dir / "rapid_link_attributes.csv").exists()
    assert (prep_dir / "kfc.csv").exists()
    assert (prep_dir / "inflow.nc").exists()


def test_run_prepared_experiment_writes_qout(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path)
    prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    run_registry = run_prepared_experiment(experiment_dir)
    assert run_registry.loc[0, "status"] == "ran"

    qout_path = Path(run_registry.loc[0, "qout_nc"])
    assert qout_path.exists()
    with netcdf_file(qout_path, "r", mmap=False) as ds:
        qout = ds.variables["Qout"].data.copy()
        assert qout.shape == (3, 1)


def test_prepare_experiment_accepts_real_link_id_zero(tmp_path: Path) -> None:
    experiment_dir = _write_state(tmp_path, link_id=0)
    registry = prepare_experiment(
        experiment_dir,
        forcing_path=experiment_dir / "forcing.csv",
        prep_config=RapidPrepConfig(),
    )
    assert registry.loc[0, "status"] == "prepared"
