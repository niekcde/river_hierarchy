"""Refresh an island setup after a direction-corrected independent rebuild.

The corrected setup redetects candidates and anchors from the new independent
hierarchy, but transfers only the prior human review decision after proving
that every island ID, component size, and geometry is unchanged.  It then
recomputes the derived equivalence registry against the corrected states.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network_variants.island_experiment import build_island_equivalence_registry
from network_variants.island_setup import create_reviewed_island_setup


def _read_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path, low_memory=False)
    required = {"island_id", "island_component_id", "n_pixels", "selection_status"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"Island candidate registry is missing fields: {missing}: {path}")
    if registry["island_id"].astype(str).duplicated().any():
        raise ValueError(f"Island candidate registry has duplicate island IDs: {path}")
    registry["island_id"] = registry["island_id"].astype(str)
    return registry


def _validate_same_reviewed_islands(
    historical_registry: pd.DataFrame,
    historical_geometries: gpd.GeoDataFrame,
    refreshed_registry: pd.DataFrame,
    refreshed_geometries: gpd.GeoDataFrame,
) -> None:
    if set(historical_registry["island_id"]) != set(refreshed_registry["island_id"]):
        raise ValueError("Direction-corrected setup changed the island-ID set; review decisions cannot be transferred.")
    historical = historical_registry.set_index("island_id")
    refreshed = refreshed_registry.set_index("island_id")
    for island_id in sorted(historical.index):
        if int(historical.at[island_id, "island_component_id"]) != int(refreshed.at[island_id, "island_component_id"]):
            raise ValueError(f"Island {island_id} has a different component ID after correction.")
        if int(historical.at[island_id, "n_pixels"]) != int(refreshed.at[island_id, "n_pixels"]):
            raise ValueError(f"Island {island_id} has a different pixel count after correction.")
    historical_geometry = historical_geometries.assign(island_id=historical_geometries["island_id"].astype(str)).set_index("island_id").geometry
    refreshed_geometry = refreshed_geometries.assign(island_id=refreshed_geometries["island_id"].astype(str)).set_index("island_id").geometry
    for island_id in sorted(historical.index):
        if not historical_geometry.loc[island_id].equals(refreshed_geometry.loc[island_id]):
            raise ValueError(f"Island {island_id} geometry changed after correction.")


def refresh_direction_corrected_island_setup(
    corrected_experiment_dir: str | Path,
    historical_setup_dir: str | Path,
    *,
    setup_dir: str | Path,
    tiny_island_pixels: int = 9,
    nearby_link_buffer_m: float = 500.0,
) -> dict[str, object]:
    """Create a corrected setup and transfer only verified review statuses."""
    corrected = Path(corrected_experiment_dir).expanduser().resolve()
    historical = Path(historical_setup_dir).expanduser().resolve()
    destination = Path(setup_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Corrected island setup destination is non-empty: {destination}")

    historical_registry = _read_registry(historical / "island_candidate_registry.csv")
    historical_geometries = gpd.read_file(historical / "island_candidate_geometries.gpkg")

    created = create_reviewed_island_setup(
        corrected,
        setup_dir=destination,
        tiny_island_pixels=tiny_island_pixels,
        nearby_link_buffer_m=nearby_link_buffer_m,
    )
    refreshed_registry_path = destination / "island_candidate_registry.csv"
    refreshed_geometries_path = destination / "island_candidate_geometries.gpkg"
    refreshed_registry = _read_registry(refreshed_registry_path)
    refreshed_geometries = gpd.read_file(refreshed_geometries_path)
    _validate_same_reviewed_islands(
        historical_registry,
        historical_geometries,
        refreshed_registry,
        refreshed_geometries,
    )

    historical_status = historical_registry.set_index("island_id")["selection_status"]
    refreshed_registry["selection_status"] = refreshed_registry["island_id"].map(historical_status)
    if refreshed_registry["selection_status"].isna().any():
        raise ValueError("A reviewed island selection status could not be transferred.")
    refreshed_registry.to_csv(refreshed_registry_path, index=False)

    # Rebuild the geometry table from the refreshed registry, ensuring its
    # descriptive fields cannot retain a stale selection_status value.
    geometry_only = refreshed_geometries[["island_id", "geometry"]].copy()
    geometry_only["island_id"] = geometry_only["island_id"].astype(str)
    refreshed_geometries_path.unlink(missing_ok=True)
    transferred_geometries = geometry_only.merge(
        refreshed_registry,
        on="island_id",
        how="inner",
        validate="1:1",
    )
    gpd.GeoDataFrame(transferred_geometries, geometry="geometry", crs=refreshed_geometries.crs).to_file(
        refreshed_geometries_path,
        driver="GPKG",
    )

    manifest_path = destination / "island_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    approved = refreshed_registry.loc[
        refreshed_registry["selection_status"].eq("approved_pilot"), "island_id"
    ].astype(str).tolist()
    manifest.update(
        {
            "approved_island_ids": approved,
            "excluded_island_ids": refreshed_registry.loc[
                refreshed_registry["selection_status"].eq("exclude_raster_artifact"), "island_id"
            ].astype(str).tolist(),
            "requires_anchor_review_island_ids": refreshed_registry.loc[
                refreshed_registry["selection_status"].eq("requires_anchor_review"), "island_id"
            ].astype(str).tolist(),
            "n_approved_islands": int(len(approved)),
            "n_tiny_excluded": int(refreshed_registry["selection_status"].eq("exclude_raster_artifact").sum()),
            "n_requires_anchor_review": int(refreshed_registry["selection_status"].eq("requires_anchor_review").sum()),
            "review_selection_transfer": {
                "historical_setup_dir": str(historical),
                "rule": "transfer selection_status only after exact island ID/component/pixel/geometry validation",
                "historical_status_counts": historical_registry["selection_status"].value_counts().to_dict(),
                "refreshed_auto_status_counts_before_transfer": created.candidate_registry["selection_status"].value_counts().to_dict(),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    equivalence_path = destination / "island_equivalence_registry.csv"
    equivalence = build_island_equivalence_registry(
        setup_dir=destination,
        experiment_dir=corrected,
        output_path=equivalence_path,
    )
    return {
        "corrected_experiment_dir": str(corrected),
        "historical_setup_dir": str(historical),
        "setup_dir": str(destination),
        "approved_islands": int(len(approved)),
        "aliases": int(equivalence["equivalence_status"].eq("alias_existing_unit_state").sum()),
        "materialized_islands": int(equivalence["equivalence_status"].eq("requires_full_island_state").sum()),
        "equivalence_registry": str(equivalence_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrected_experiment_dir")
    parser.add_argument("historical_setup_dir")
    parser.add_argument("--setup-dir", required=True)
    parser.add_argument("--tiny-island-pixels", type=int, default=9)
    parser.add_argument("--nearby-link-buffer-m", type=float, default=500.0)
    args = parser.parse_args(argv)
    result = refresh_direction_corrected_island_setup(
        args.corrected_experiment_dir,
        args.historical_setup_dir,
        setup_dir=args.setup_dir,
        tiny_island_pixels=args.tiny_island_pixels,
        nearby_link_buffer_m=args.nearby_link_buffer_m,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
