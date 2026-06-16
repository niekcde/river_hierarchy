from __future__ import annotations

import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "rivgraph_centerline" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rivgraph_centerline.workflow import WorkflowPaths, _clear_rivgraph_outputs, run_rivgraph_for_run


def test_clear_rivgraph_outputs_removes_only_run_prefixed_files(tmp_path) -> None:
    paths = WorkflowPaths.from_name("02", tmp_path / "run")
    paths.ensure_dirs()

    stale_fixlinks = paths.rivgraph_dir / "02_fixlinks.csv"
    stale_links = paths.rivgraph_dir / "02_links.gpkg"
    unrelated = paths.rivgraph_dir / "notes.txt"
    nested = paths.rivgraph_dir / "02_debug"
    nested_file = nested / "payload.txt"

    stale_fixlinks.write_text("old", encoding="utf-8")
    stale_links.write_text("old", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")
    nested.mkdir()
    nested_file.write_text("nested", encoding="utf-8")

    deleted = _clear_rivgraph_outputs(paths)

    assert str(stale_fixlinks) in deleted
    assert str(stale_links) in deleted
    assert str(nested) in deleted
    assert not stale_fixlinks.exists()
    assert not stale_links.exists()
    assert not nested.exists()
    assert unrelated.exists()


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
    assert summary["outputs"]["manual_fixlinks_csv"] == str(paths.rivgraph_dir / "02_fixlinks.csv")
