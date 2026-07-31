from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from hierarchy_level_definition.graph_building.directed_network_checks import (
    analyze_reviewed_network,
    write_outputs as write_validation_outputs,
)


DEFAULT_REQUEST_COLUMNS = ("example_id", "link_id")
DEFAULT_LINKS_NAME = "directed_links.gpkg"
DEFAULT_NODES_NAME = "reviewed_nodes.gpkg"
DEFAULT_LINK_CSV_NAME = "manual_link_directions.csv"


@dataclass(frozen=True)
class ReviewPaths:
    example_id: str
    review_dir: Path
    links_path: Path
    nodes_path: Path
    manual_link_csv_path: Path
    validation_output_dir: Path
    layout: str


def parse_id_nodes(value: Any) -> tuple[int, int]:
    if value is None or pd.isna(value):
        raise ValueError("Encountered missing id_nodes value.")
    parts = [part.strip() for part in str(value).replace("[", "").replace("]", "").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two nodes in id_nodes, got {value!r}.")
    return int(parts[0]), int(parts[1])


def as_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def reverse_linear_geometry(geometry: Any) -> Any:
    if geometry is None or geometry.is_empty:
        return geometry
    if isinstance(geometry, LineString):
        return LineString(list(geometry.coords)[::-1])
    if isinstance(geometry, MultiLineString):
        reversed_parts = [LineString(list(part.coords)[::-1]) for part in reversed(list(geometry.geoms))]
        return MultiLineString(reversed_parts)
    if hasattr(geometry, "reverse"):
        return geometry.reverse()
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def geometry_endpoints(geometry: Any) -> tuple[Point, Point]:
    if geometry is None or geometry.is_empty:
        raise ValueError("Cannot determine endpoints for empty geometry.")
    if isinstance(geometry, LineString):
        coords = list(geometry.coords)
        return Point(coords[0]), Point(coords[-1])
    if isinstance(geometry, MultiLineString):
        parts = list(geometry.geoms)
        first_coords = list(parts[0].coords)
        last_coords = list(parts[-1].coords)
        return Point(first_coords[0]), Point(last_coords[-1])
    if hasattr(geometry, "boundary") and geometry.boundary is not None:
        boundary = geometry.boundary
        if hasattr(boundary, "geoms") and len(boundary.geoms) >= 2:
            return boundary.geoms[0], boundary.geoms[-1]
    raise TypeError(f"Unsupported linear geometry type: {geometry.geom_type}")


def geometry_node_order(
    geometry: Any,
    node_a: int,
    node_b: int,
    node_geom_a: Any,
    node_geom_b: Any,
) -> tuple[int, int]:
    start_point, end_point = geometry_endpoints(geometry)
    score_ab = start_point.distance(node_geom_a) + end_point.distance(node_geom_b)
    score_ba = start_point.distance(node_geom_b) + end_point.distance(node_geom_a)
    if score_ab <= score_ba:
        return node_a, node_b
    return node_b, node_a


def backup_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_old{path.suffix}")


def resolve_review_paths(
    example_id: str,
    *,
    review_root: str | Path,
    validation_root: str | Path,
) -> ReviewPaths:
    base_dir = Path(review_root) / example_id
    validation_output_dir = Path(validation_root) / example_id

    manual_review_links = base_dir / DEFAULT_LINKS_NAME
    manual_review_nodes = base_dir / DEFAULT_NODES_NAME
    if manual_review_links.exists() and manual_review_nodes.exists():
        return ReviewPaths(
            example_id=example_id,
            review_dir=base_dir,
            links_path=manual_review_links,
            nodes_path=manual_review_nodes,
            manual_link_csv_path=base_dir / DEFAULT_LINK_CSV_NAME,
            validation_output_dir=validation_output_dir,
            layout="manual_review",
        )

    rivgraph_dir = base_dir / "rivgraph"
    rivgraph_links = rivgraph_dir / f"{example_id}_links.gpkg"
    rivgraph_nodes = rivgraph_dir / f"{example_id}_nodes.gpkg"
    if rivgraph_links.exists() and rivgraph_nodes.exists():
        return ReviewPaths(
            example_id=example_id,
            review_dir=rivgraph_dir,
            links_path=rivgraph_links,
            nodes_path=rivgraph_nodes,
            manual_link_csv_path=rivgraph_dir / DEFAULT_LINK_CSV_NAME,
            validation_output_dir=validation_output_dir,
            layout="rivgraph_run",
        )

    return ReviewPaths(
        example_id=example_id,
        review_dir=base_dir,
        links_path=manual_review_links,
        nodes_path=manual_review_nodes,
        manual_link_csv_path=base_dir / DEFAULT_LINK_CSV_NAME,
        validation_output_dir=validation_output_dir,
        layout="unresolved",
    )


def load_requests(csv_path: str | Path) -> pd.DataFrame:
    requests = pd.read_csv(csv_path, sep=None, engine="python")
    missing = [column for column in DEFAULT_REQUEST_COLUMNS if column not in requests.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in reversal CSV: {missing}.")

    requests = requests.copy()
    requests = requests.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    requests["example_id"] = requests["example_id"].astype(str).str.strip().map(normalize_example_id)
    requests["link_id"] = requests["link_id"].astype("Int64")

    if requests["example_id"].eq("").any():
        raise ValueError("Found blank example_id value(s) in reversal CSV.")
    if requests["link_id"].isna().any():
        raise ValueError("Found missing or non-integer link_id value(s) in reversal CSV.")

    duplicate_mask = requests.duplicated(subset=["example_id", "link_id"], keep=False)
    if duplicate_mask.any():
        duplicates = (
            requests.loc[duplicate_mask, ["example_id", "link_id"]]
            .sort_values(["example_id", "link_id"], kind="mergesort")
            .to_dict(orient="records")
        )
        raise ValueError(f"Found duplicate example_id/link_id reversal requests: {duplicates}.")

    requests["link_id"] = requests["link_id"].astype(int)
    return requests


def normalize_example_id(value: Any) -> str:
    text = str(value).strip()
    if text == "":
        return text
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return f"{int(numeric):02d}"
    return text


def ensure_exists(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def copy_backup(path: Path, *, overwrite: bool) -> Path:
    destination = backup_path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Backup already exists: {destination}")
    shutil.copy2(path, destination)
    return destination


def build_manual_direction_frame(
    links: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> pd.DataFrame:
    node_geom = dict(zip(nodes["id_node"].astype(int), nodes.geometry))
    records: list[dict[str, Any]] = []

    for row in links.itertuples(index=False):
        link_id = int(row.id_link)
        node_a, node_b = parse_id_nodes(row.id_nodes)
        usnode = int(getattr(row, "id_us_node", node_a))
        dsnode = int(getattr(row, "id_ds_node", node_b))
        geometry_start, geometry_end = geometry_node_order(
            row.geometry,
            node_a,
            node_b,
            node_geom[node_a],
            node_geom[node_b],
        )
        geometry_reversed = usnode != geometry_start
        records.append(
            {
                "link_id": link_id,
                "usnode": usnode,
                "dsnode": dsnode,
                "source_node_a": node_a,
                "source_node_b": node_b,
                "source_geometry_start_node": geometry_start,
                "source_geometry_end_node": geometry_end,
                "geometry_reversed": geometry_reversed,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("link_id").reset_index(drop=True)


def update_manual_direction_frame(
    manual_link_frame: pd.DataFrame,
    reversed_link_ids: set[int],
) -> pd.DataFrame:
    frame = manual_link_frame.copy()
    if "link_id" not in frame.columns:
        raise ValueError("manual_link_directions.csv is missing required column 'link_id'.")

    frame["link_id"] = frame["link_id"].astype(int)
    missing_rows = sorted(reversed_link_ids.difference(set(frame["link_id"])))
    if missing_rows:
        raise ValueError(
            "manual_link_directions.csv is missing requested link_id row(s): "
            f"{missing_rows}."
        )

    for link_id in sorted(reversed_link_ids):
        index = frame.index[frame["link_id"] == link_id]
        if len(index) != 1:
            raise ValueError(f"Expected exactly one manual_link_directions.csv row for link_id {link_id}.")
        row_index = index[0]
        if {"usnode", "dsnode"}.issubset(frame.columns):
            usnode = int(frame.at[row_index, "usnode"])
            dsnode = int(frame.at[row_index, "dsnode"])
            frame.at[row_index, "usnode"] = dsnode
            frame.at[row_index, "dsnode"] = usnode
        if "geometry_reversed" in frame.columns:
            frame.at[row_index, "geometry_reversed"] = not as_bool(frame.at[row_index, "geometry_reversed"])

    return frame.sort_values("link_id", kind="mergesort").reset_index(drop=True)


def apply_reversals(
    links: gpd.GeoDataFrame,
    reversed_link_ids: set[int],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    edited = links.copy()
    edited["id_link"] = edited["id_link"].astype(int)
    link_index = dict(zip(edited["id_link"], edited.index))

    missing_link_ids = sorted(reversed_link_ids.difference(set(link_index)))
    if missing_link_ids:
        raise ValueError(f"Requested link_id(s) not present in reviewed links file: {missing_link_ids}.")

    reversal_records: list[dict[str, Any]] = []
    for link_id in sorted(reversed_link_ids):
        row_index = link_index[link_id]
        old_node_a, old_node_b = parse_id_nodes(edited.at[row_index, "id_nodes"])
        old_us = int(edited.at[row_index, "id_us_node"]) if "id_us_node" in edited.columns else old_node_a
        old_ds = int(edited.at[row_index, "id_ds_node"]) if "id_ds_node" in edited.columns else old_node_b
        new_us = old_ds
        new_ds = old_us

        edited.at[row_index, "id_nodes"] = f"{new_us}, {new_ds}"
        if "id_us_node" in edited.columns:
            edited.at[row_index, "id_us_node"] = new_us
        if "id_ds_node" in edited.columns:
            edited.at[row_index, "id_ds_node"] = new_ds
        if "manual_direction_reviewed" in edited.columns:
            edited.at[row_index, "manual_direction_reviewed"] = True
        if "manual_us_node" in edited.columns:
            edited.at[row_index, "manual_us_node"] = new_us
        if "manual_ds_node" in edited.columns:
            edited.at[row_index, "manual_ds_node"] = new_ds
        if "geometry_reversed_manual" in edited.columns:
            edited.at[row_index, "geometry_reversed_manual"] = not as_bool(edited.at[row_index, "geometry_reversed_manual"])
        edited.at[row_index, edited.geometry.name] = reverse_linear_geometry(edited.at[row_index, edited.geometry.name])

        reversal_records.append(
            {
                "link_id": link_id,
                "old_us_node": old_us,
                "old_ds_node": old_ds,
                "new_us_node": new_us,
                "new_ds_node": new_ds,
                "old_id_nodes": f"{old_node_a}, {old_node_b}",
                "new_id_nodes": f"{new_us}, {new_ds}",
            }
        )

    return edited, pd.DataFrame.from_records(reversal_records)


def write_review_outputs(
    paths: ReviewPaths,
    links: gpd.GeoDataFrame,
    manual_link_frame: pd.DataFrame,
) -> None:
    links.to_file(paths.links_path, driver="GPKG")
    manual_link_frame.to_csv(paths.manual_link_csv_path, index=False)


def run_validation(paths: ReviewPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    _, degree_frame, report = analyze_reviewed_network(paths.links_path, paths.nodes_path)
    write_validation_outputs(paths.validation_output_dir, degree_frame, report)
    return degree_frame, report.to_dict()


def process_example(
    example_id: str,
    requested_links: list[int],
    *,
    review_root: str | Path,
    validation_root: str | Path,
    old_backup_mode: Literal["error", "overwrite", "use"] = "error",
) -> dict[str, Any]:
    paths = resolve_review_paths(
        example_id,
        review_root=review_root,
        validation_root=validation_root,
    )
    ensure_exists(paths.links_path, label="Reviewed links GeoPackage")
    ensure_exists(paths.nodes_path, label="Reviewed nodes GeoPackage")

    if old_backup_mode not in {"error", "overwrite", "use"}:
        raise ValueError(f"Unsupported old backup mode: {old_backup_mode!r}")

    if old_backup_mode == "use":
        links_backup = backup_path(paths.links_path)
        manual_link_backup = backup_path(paths.manual_link_csv_path)
        if not links_backup.exists():
            copy_backup(paths.links_path, overwrite=False)
        if paths.manual_link_csv_path.exists() and not manual_link_backup.exists():
            copy_backup(paths.manual_link_csv_path, overwrite=False)
        links_source_path = links_backup
        manual_link_source_path = manual_link_backup if manual_link_backup.exists() else None
    else:
        links_backup = copy_backup(paths.links_path, overwrite=old_backup_mode == "overwrite")
        manual_link_backup = None
        if paths.manual_link_csv_path.exists():
            manual_link_backup = copy_backup(paths.manual_link_csv_path, overwrite=old_backup_mode == "overwrite")
        links_source_path = paths.links_path
        manual_link_source_path = paths.manual_link_csv_path if paths.manual_link_csv_path.exists() else None

    links = gpd.read_file(links_source_path)
    nodes = gpd.read_file(paths.nodes_path)
    reversed_link_ids = {int(link_id) for link_id in requested_links}
    edited_links, reversal_frame = apply_reversals(links, reversed_link_ids)

    if manual_link_source_path is not None:
        manual_link_frame = update_manual_direction_frame(pd.read_csv(manual_link_source_path), reversed_link_ids)
    else:
        manual_link_frame = build_manual_direction_frame(edited_links, nodes)

    write_review_outputs(paths, edited_links, manual_link_frame)
    _, validation_report = run_validation(paths)

    return {
        "example_id": example_id,
        "n_links_reversed": len(reversed_link_ids),
        "reversed_link_ids": sorted(reversed_link_ids),
        "links_path": str(paths.links_path),
        "links_source_path": str(links_source_path),
        "old_backup_mode": old_backup_mode,
        "nodes_path": str(paths.nodes_path),
        "links_backup_path": str(links_backup),
        "manual_link_csv_path": str(paths.manual_link_csv_path),
        "manual_link_csv_backup_path": str(manual_link_backup) if manual_link_backup is not None and manual_link_backup.exists() else None,
        "validation_output_dir": str(paths.validation_output_dir),
        "validation_is_valid": bool(validation_report["is_valid"]),
        "validation_issues": validation_report["issues"],
        "reversal_details": reversal_frame.to_dict(orient="records"),
    }


def default_review_root() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def default_validation_root() -> Path:
    return Path(__file__).resolve().parents[1] / "graph_building" / "outputs"


def default_summary_path(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_summary.json")


def filter_requests_to_example(
    requests: pd.DataFrame,
    *,
    example_id: str | None,
) -> pd.DataFrame:
    if example_id is None:
        return requests

    normalized = normalize_example_id(example_id)
    filtered = requests.loc[requests["example_id"] == normalized].copy()
    if filtered.empty:
        available = sorted(requests["example_id"].astype(str).unique().tolist())
        raise ValueError(
            f"No reversal requests found for example_id {example_id!r} "
            f"(normalized to {normalized!r}). Available example_id values: {available}."
        )
    return filtered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply per-link manual reversals to reviewed directed_links.gpkg files, overwrite "
            "the working review file after writing *_old backups, and immediately rerun the "
            "same directed-network checks used after GUI review."
        )
    )
    parser.add_argument("--csv", required=True, help="CSV with at least example_id and link_id columns.")
    parser.add_argument(
        "--example-id",
        default=None,
        help="Optional single example_id filter. Values like 2 and 02 are treated equivalently.",
    )
    parser.add_argument(
        "--review-root",
        default=str(default_review_root()),
        help="Root directory containing manual_review/outputs/<example_id>/ reviewed files.",
    )
    parser.add_argument(
        "--validation-root",
        default=str(default_validation_root()),
        help="Root directory for graph_building validation outputs, written to <validation-root>/<example_id>/.",
    )
    parser.add_argument(
        "--old-backup-mode",
        choices=("error", "overwrite", "use"),
        default="error",
        help=(
            "How to handle *_old files: error (default) refuses an existing backup; "
            "overwrite replaces it from the current working files; use reads it as the "
            "immutable baseline (creating it from the current working files if absent) "
            "and regenerates the current working files."
        ),
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Defaults to <csv>_summary.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    requests = load_requests(args.csv)
    requests = filter_requests_to_example(requests, example_id=args.example_id)
    grouped = requests.groupby("example_id", sort=True)["link_id"].agg(list)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for example_id, link_ids in grouped.items():
        try:
            result = process_example(
                example_id,
                link_ids,
                review_root=args.review_root,
                validation_root=args.validation_root,
                old_backup_mode=args.old_backup_mode,
            )
            results.append(result)
            print(
                f"{example_id}: reversed {result['n_links_reversed']} link(s), "
                f"validation_is_valid={result['validation_is_valid']}"
            )
            if result["validation_issues"]:
                for issue in result["validation_issues"]:
                    print(f"  - {issue}")
                failures.append(result)
        except Exception as exc:
            failure = {
                "example_id": example_id,
                "error": str(exc),
            }
            failures.append(failure)
            results.append(failure)
            print(f"{example_id}: ERROR {exc}", file=sys.stderr)

    summary_path = Path(args.summary_json) if args.summary_json is not None else default_summary_path(args.csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote summary to {summary_path.resolve()}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
