from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPoint, Point, Polygon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.raster_mask import crop_binary_mask
from rivgraph_centerline.workflow import (
    WorkflowPaths,
    _clear_rivgraph_outputs,
    _patched_rivgraph_dir_centerline,
    _representative_project_point,
    prepare_binary_projected_mask,
    run_rivgraph_for_run,
)
from rivgraph_compat import sanitize_centerline_link_fields


def test_crop_binary_mask_reduces_empty_margins_and_updates_transform() -> None:
    mask = np.zeros((8, 10), dtype="uint8")
    mask[3:5, 4:7] = 1
    transform = from_origin(100, 200, 5, 5)

    cropped, cropped_transform, crop_window = crop_binary_mask(mask, transform, padding=1)

    assert cropped.shape == (4, 5)
    assert int(cropped.sum()) == 6
    assert crop_window == {"row_min": 2, "row_max": 6, "col_min": 3, "col_max": 8}
    assert cropped_transform.c == 115
    assert cropped_transform.f == 190


def test_prepare_binary_projected_mask_trims_projected_output(tmp_path) -> None:
    src_path = tmp_path / "source.tif"
    dst_path = tmp_path / "prepared.tif"
    transform = from_origin(0, 10, 1, 1)
    data = np.zeros((10, 12), dtype="uint8")
    data[4:6, 5:7] = 1

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:32631",
        "transform": transform,
        "nodata": 0,
    }
    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(data, 1)

    summary = prepare_binary_projected_mask(src_path=src_path, dst_path=dst_path, threshold=0.5, dst_crs="EPSG:32631")

    with rasterio.open(dst_path) as prepared:
        prepared_data = prepared.read(1)
        assert prepared_data.shape == (4, 4)
        assert int(prepared_data.sum()) == 4
        assert prepared.transform.c == 4
        assert prepared.transform.f == 7

    assert summary["source_shape"] == [10, 12]
    assert summary["prepared_shape"] == [4, 4]


def test_clear_rivgraph_outputs_removes_only_run_prefixed_files(tmp_path) -> None:
    paths = WorkflowPaths.from_name("02", tmp_path / "run")
    paths.ensure_dirs()

    stale_fixlinks = paths.rivgraph_dir / "02_fixlinks.csv"
    stale_links = paths.rivgraph_dir / "02_links.gpkg"
    unrelated = paths.rivgraph_dir / "notes.txt"
    nested = paths.rivgraph_dir / "02_debug"
    nested_file = nested / "payload.txt"
    permanent_review = paths.run_dir / "width_review" / "base_width_review_decisions.csv"

    stale_fixlinks.write_text("old", encoding="utf-8")
    stale_links.write_text("old", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    nested.mkdir()
    nested_file.write_text("nested", encoding="utf-8")
    permanent_review.parent.mkdir(parents=True)
    permanent_review.write_text("id_link,review_decision\n1,good\n", encoding="utf-8")

    deleted = _clear_rivgraph_outputs(paths)

    assert str(stale_fixlinks) in deleted
    assert str(stale_links) in deleted
    assert str(nested) in deleted
    assert not stale_fixlinks.exists()
    assert not stale_links.exists()
    assert not nested.exists()
    assert unrelated.exists()
    assert permanent_review.exists()


def test_sanitize_rivgraph_centerline_link_fields_drops_scalar_metadata() -> None:
    links = {
        "id": [0, 1, 2],
        "idx": [[1], [2], [3]],
        "conn": [[0, 1], [1, 2], [2, 3]],
        "len": [1.0, 2.0, 3.0],
        "n_networks": 1,
        "_ids_finalized": False,
        "_ids_finalization_method": "topology not finalized",
        "bad_scalar": 7,
        "bad_text": "abc",
        "bad_short": [1, 2],
    }

    sanitize_centerline_link_fields(links)

    assert set(links) == {"id", "idx", "conn", "len", "n_networks"}


def test_representative_project_point_reduces_non_point_geometries() -> None:
    multipoint = MultiPoint([(0, 0), (1, 1)])
    multiline = MultiLineString([[(0, 0), (4, 0)], [(0, 0), (1, 0)]])
    geom_collection = GeometryCollection([LineString([(0, 0), (2, 0)]), Point(3, 3)])

    reduced_multipoint = _representative_project_point(multipoint)
    reduced_multiline = _representative_project_point(multiline)
    reduced_collection = _representative_project_point(geom_collection)

    assert isinstance(reduced_multipoint, Point)
    assert tuple(reduced_multipoint.coords)[0] == (0.0, 0.0)
    assert isinstance(reduced_multiline, Point)
    assert reduced_multiline.equals(Point(2, 0))
    assert isinstance(reduced_collection, Point)
    assert reduced_collection.equals(Point(3, 3))


def test_patched_rivgraph_dir_centerline_averages_ambiguous_mesh_membership() -> None:
    class DummyDy:
        @staticmethod
        def algmap(name: str) -> str:
            return name

        @staticmethod
        def get_link_vector(links, nodes, link_id, imshape, pixlen=1.0):
            return np.array([1.0, 0.0])

    class DummyGu:
        @staticmethod
        def xy_to_coords(cols, rows, gt):
            return np.asarray(cols, dtype=float), np.asarray(rows, dtype=float)

    def original_dir_centerline(*args, **kwargs):
        raise AssertionError("patched wrapper should be used instead")

    original_dir_centerline.__globals__.update(
        {
            "gpd": gpd,
            "np": np,
            "Point": Point,
            "Polygon": Polygon,
            "gu": DummyGu,
            "dy": DummyDy,
            "math": __import__("math"),
        }
    )

    links = {
        "id": [10],
        "conn": [[1, 2]],
        "idx": [[0, 1]],
        "guess": [[]],
        "guess_alg": [[]],
    }
    nodes = {"id": [1, 2], "idx": [0, 1]}
    meshpolys = [
        Polygon([(0, -1), (1, -1), (1, 1), (0, 1)]),
        Polygon([(1, -1), (2, -1), (2, 1), (1, 1)]),
        Polygon([(2, -1), (3, -1), (3, 1), (2, 1)]),
    ]
    meshlines = [
        LineString([(0, -1), (0, 1)]),
        LineString([(1, -1), (1, 1)]),
        LineString([(2, -1), (2, 1)]),
        LineString([(3, -1), (3, 1)]),
    ]
    imask = np.zeros((3, 3), dtype=int)
    gt = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    result = _patched_rivgraph_dir_centerline(
        original_dir_centerline,
        links,
        nodes,
        meshpolys,
        meshlines,
        imask,
        gt,
        1.0,
    )

    assert np.isclose(result["cldists"][0, 0], 0.5)
    assert result["guess"][0] == [1, 1]
    assert result["guess_alg"][0] == ["cl_dist_guess", "cl_ang_guess"]
    assert np.isfinite(result["clangs"][0, 0])


def test_run_rivgraph_for_run_cleans_stale_outputs_before_rerun(tmp_path, monkeypatch) -> None:
    paths = WorkflowPaths.from_name("02", tmp_path / "run")
    paths.ensure_dirs()

    mask_path = tmp_path / "mask.tif"
    mask_path.write_text("mask", encoding="utf-8")

    stale_fixlinks = paths.rivgraph_dir / "02_fixlinks.csv"
    stale_fixlinks.write_text("stale-fixlinks", encoding="utf-8")
    unrelated = paths.rivgraph_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    class DummyRiver:
        def __init__(self, *, name: str, path_to_mask: str, results_folder: str, exit_sides: str, verbose: bool) -> None:
            self.name = name
            self.path_to_mask = path_to_mask
            self.results_folder = Path(results_folder)
            self.exit_sides = exit_sides
            self.verbose = verbose
            self.Iskel = "iskel"
            self.links = {"id": [0, 1], "certain": [1, 1]}
            self.nodes = {"id": [0, 1], "inlets": [1], "outlets": [0]}
            self.paths = {
                "Iskel": str(self.results_folder / f"{self.name}_skel.tif"),
                "links": str(self.results_folder / f"{self.name}_links.gpkg"),
                "nodes": str(self.results_folder / f"{self.name}_nodes.gpkg"),
                "centerline": str(self.results_folder / f"{self.name}_centerline.gpkg"),
                "linkdirs": str(self.results_folder / f"{self.name}_link_directions.tif"),
                "fixlinks_csv": str(self.results_folder / f"{self.name}_fixlinks.csv"),
                "log": str(self.results_folder / f"{self.name}_log.log"),
            }

        def skeletonize(self) -> None:
            return None

        def to_geotiff(self, kind: str) -> None:
            target = self.paths["Iskel"] if kind == "skeleton" else self.paths["linkdirs"]
            Path(target).write_text(kind, encoding="utf-8")

        def compute_network(self) -> None:
            return None

        def finalize_ids(self) -> None:
            return None

        def compute_link_width_and_length(self) -> None:
            return None

        def compute_centerline(self) -> None:
            return None

        def assign_flow_directions(self) -> None:
            Path(self.paths["fixlinks_csv"]).write_text("fresh-fixlinks", encoding="utf-8")

        def to_geovectors(self, kind: str, ftype: str = "gpkg") -> None:
            target = self.paths["centerline"] if kind == "centerline" else self.paths["links"]
            if kind == "network":
                Path(self.paths["links"]).write_text("links", encoding="utf-8")
                Path(self.paths["nodes"]).write_text("nodes", encoding="utf-8")
            else:
                Path(target).write_text(ftype, encoding="utf-8")

    rivgraph_module = types.ModuleType("rivgraph")
    classes_module = types.ModuleType("rivgraph.classes")
    rivers_module = types.ModuleType("rivgraph.rivers")
    river_utils_module = types.ModuleType("rivgraph.rivers.river_utils")

    classes_module.river = DummyRiver
    river_utils_module.prune_river = lambda links, nodes, exit_sides, iskel: (links, nodes)
    rivgraph_module.classes = classes_module
    rivgraph_module.rivers = rivers_module
    rivers_module.river_utils = river_utils_module

    monkeypatch.setitem(sys.modules, "rivgraph", rivgraph_module)
    monkeypatch.setitem(sys.modules, "rivgraph.classes", classes_module)
    monkeypatch.setitem(sys.modules, "rivgraph.rivers", rivers_module)
    monkeypatch.setitem(sys.modules, "rivgraph.rivers.river_utils", river_utils_module)

    _, summary = run_rivgraph_for_run(
        name="02",
        run_dir=paths.run_dir,
        mask=mask_path,
        exit_sides="SN",
        assign_flow=True,
        verbose=False,
    )

    assert summary["cleaned_rivgraph_target"] is True
    assert str(stale_fixlinks) in summary["deleted_rivgraph_outputs"]
    assert stale_fixlinks.read_text(encoding="utf-8") == "fresh-fixlinks"
    assert unrelated.exists()
    assert summary["outputs"]["manual_fixlinks_csv"] == str(stale_fixlinks)


def test_run_rivgraph_for_run_falls_back_to_centerline_mesh_for_flow_assignment(tmp_path, monkeypatch) -> None:
    paths = WorkflowPaths.from_name("02", tmp_path / "run")
    paths.ensure_dirs()

    mask_path = tmp_path / "mask.tif"
    mask_path.write_text("mask", encoding="utf-8")

    class DummyRiver:
        def __init__(self, *, name: str, path_to_mask: str, results_folder: str, exit_sides: str, verbose: bool) -> None:
            self.name = name
            self.path_to_mask = path_to_mask
            self.results_folder = Path(results_folder)
            self.exit_sides = exit_sides
            self.verbose = verbose
            self.single_thread = False
            self.Iskel = "iskel"
            self.links = {"id": [0, 1], "certain": [1, 1]}
            self.nodes = {"id": [0, 1], "inlets": [1], "outlets": [0]}
            self.paths = {
                "Iskel": str(self.results_folder / f"{self.name}_skel.tif"),
                "links": str(self.results_folder / f"{self.name}_links.gpkg"),
                "nodes": str(self.results_folder / f"{self.name}_nodes.gpkg"),
                "centerline": str(self.results_folder / f"{self.name}_centerline.gpkg"),
                "linkdirs": str(self.results_folder / f"{self.name}_link_directions.tif"),
                "fixlinks_csv": str(self.results_folder / f"{self.name}_fixlinks.csv"),
                "log": str(self.results_folder / f"{self.name}_log.log"),
            }

        def skeletonize(self) -> None:
            return None

        def to_geotiff(self, kind: str) -> None:
            target = self.paths["Iskel"] if kind == "skeleton" else self.paths["linkdirs"]
            Path(target).write_text(kind, encoding="utf-8")

        def compute_network(self) -> None:
            return None

        def finalize_ids(self) -> None:
            return None

        def compute_link_width_and_length(self) -> None:
            return None

        def compute_centerline(self) -> None:
            return None

        def compute_mesh(self, single_thread: bool = False) -> None:
            self.meshpolys = ["mesh"]
            self.meshlines = ["line"]
            self.centerline_smooth = "smooth"
            self.single_thread = single_thread

        def assign_flow_directions(self) -> None:
            if getattr(self, "single_thread", False) is False:
                raise IndexError("list index out of range")
            Path(self.paths["fixlinks_csv"]).write_text("fallback-fixlinks", encoding="utf-8")

        def to_geovectors(self, kind: str, ftype: str = "gpkg") -> None:
            target = self.paths["centerline"] if kind == "centerline" else self.paths["links"]
            if kind == "network":
                Path(self.paths["links"]).write_text("links", encoding="utf-8")
                Path(self.paths["nodes"]).write_text("nodes", encoding="utf-8")
            else:
                Path(target).write_text(ftype, encoding="utf-8")

    rivgraph_module = types.ModuleType("rivgraph")
    classes_module = types.ModuleType("rivgraph.classes")
    rivers_module = types.ModuleType("rivgraph.rivers")
    river_utils_module = types.ModuleType("rivgraph.rivers.river_utils")

    classes_module.river = DummyRiver
    river_utils_module.prune_river = lambda links, nodes, exit_sides, iskel: (links, nodes)
    rivgraph_module.classes = classes_module
    rivgraph_module.rivers = rivers_module
    rivers_module.river_utils = river_utils_module

    monkeypatch.setitem(sys.modules, "rivgraph", rivgraph_module)
    monkeypatch.setitem(sys.modules, "rivgraph.classes", classes_module)
    monkeypatch.setitem(sys.modules, "rivgraph.rivers", rivers_module)
    monkeypatch.setitem(sys.modules, "rivgraph.rivers.river_utils", river_utils_module)

    _, summary = run_rivgraph_for_run(
        name="02",
        run_dir=paths.run_dir,
        mask=mask_path,
        exit_sides="SN",
        assign_flow=True,
        verbose=False,
        flow_mesh_mode="auto",
    )

    assert summary["flow_mesh_mode_requested"] == "auto"
    assert summary["flow_mesh_mode_used"] == "centerline"
    assert summary["flow_mesh_fallback_applied"] is True
    assert "IndexError: list index out of range" == summary["flow_mesh_primary_error"]
    assert summary["flow_mesh_secondary_error"] is None
    assert summary["flow_directions_requested"] is True
    assert summary["flow_directions_assigned"] is True
    assert summary["outputs"]["manual_fixlinks_csv"] == str(paths.rivgraph_dir / "02_fixlinks.csv")


def test_run_rivgraph_for_run_records_double_mesh_failure_without_crashing(tmp_path, monkeypatch) -> None:
    paths = WorkflowPaths.from_name("02", tmp_path / "run")
    paths.ensure_dirs()

    mask_path = tmp_path / "mask.tif"
    mask_path.write_text("mask", encoding="utf-8")

    class DummyRiver:
        def __init__(self, *, name: str, path_to_mask: str, results_folder: str, exit_sides: str, verbose: bool) -> None:
            self.name = name
            self.path_to_mask = path_to_mask
            self.results_folder = Path(results_folder)
            self.exit_sides = exit_sides
            self.verbose = verbose
            self.single_thread = False
            self.Iskel = "iskel"
            self.links = {"id": [0, 1]}
            self.nodes = {"id": [0, 1], "inlets": [1], "outlets": [0]}
            self.paths = {
                "Iskel": str(self.results_folder / f"{self.name}_skel.tif"),
                "links": str(self.results_folder / f"{self.name}_links.gpkg"),
                "nodes": str(self.results_folder / f"{self.name}_nodes.gpkg"),
                "centerline": str(self.results_folder / f"{self.name}_centerline.gpkg"),
                "linkdirs": str(self.results_folder / f"{self.name}_link_directions.tif"),
                "fixlinks_csv": str(self.results_folder / f"{self.name}_fixlinks.csv"),
                "log": str(self.results_folder / f"{self.name}_log.log"),
            }

        def skeletonize(self) -> None:
            return None

        def to_geotiff(self, kind: str) -> None:
            if kind == "skeleton":
                Path(self.paths["Iskel"]).write_text(kind, encoding="utf-8")
            else:
                Path(self.paths["linkdirs"]).write_text(kind, encoding="utf-8")

        def compute_network(self) -> None:
            return None

        def finalize_ids(self) -> None:
            return None

        def compute_link_width_and_length(self) -> None:
            return None

        def compute_centerline(self) -> None:
            return None

        def compute_mesh(self, single_thread: bool = False) -> None:
            self.single_thread = single_thread
            if single_thread:
                raise IndexError("index 2 is out of bounds for axis 0 with size 1")

        def assign_flow_directions(self) -> None:
            if getattr(self, "single_thread", False) is False:
                raise IndexError("list index out of range")
            raise AssertionError("centerline fallback should fail during compute_mesh first")

        def to_geovectors(self, kind: str, ftype: str = "gpkg") -> None:
            if kind == "network":
                Path(self.paths["links"]).write_text("links", encoding="utf-8")
                Path(self.paths["nodes"]).write_text("nodes", encoding="utf-8")
            else:
                Path(self.paths["centerline"]).write_text(ftype, encoding="utf-8")

    rivgraph_module = types.ModuleType("rivgraph")
    classes_module = types.ModuleType("rivgraph.classes")
    rivers_module = types.ModuleType("rivgraph.rivers")
    river_utils_module = types.ModuleType("rivgraph.rivers.river_utils")

    classes_module.river = DummyRiver
    river_utils_module.prune_river = lambda links, nodes, exit_sides, iskel: (links, nodes)
    rivgraph_module.classes = classes_module
    rivgraph_module.rivers = rivers_module
    rivers_module.river_utils = river_utils_module

    monkeypatch.setitem(sys.modules, "rivgraph", rivgraph_module)
    monkeypatch.setitem(sys.modules, "rivgraph.classes", classes_module)
    monkeypatch.setitem(sys.modules, "rivgraph.rivers", rivers_module)
    monkeypatch.setitem(sys.modules, "rivgraph.rivers.river_utils", river_utils_module)

    _, summary = run_rivgraph_for_run(
        name="02",
        run_dir=paths.run_dir,
        mask=mask_path,
        exit_sides="SN",
        assign_flow=True,
        verbose=False,
        flow_mesh_mode="auto",
    )

    assert summary["flow_directions_requested"] is True
    assert summary["flow_directions_assigned"] is False
    assert summary["flow_mesh_mode_requested"] == "auto"
    assert summary["flow_mesh_mode_used"] is None
    assert summary["flow_mesh_fallback_applied"] is True
    assert summary["flow_mesh_primary_error"] == "IndexError: list index out of range"
    assert summary["flow_mesh_secondary_error"] == "IndexError: index 2 is out of bounds for axis 0 with size 1"
    assert summary["outputs"]["direction_raster"] is None
    assert summary["outputs"]["links"] == str(paths.rivgraph_dir / "02_links.gpkg")
    assert Path(summary["outputs"]["links"]).exists()
    assert Path(summary["outputs"]["centerline"]).exists()
