from __future__ import annotations

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from .qa_exports import export_subdaily_hierarchy_package
from .subdaily_download import download_subdaily_from_audit
from .us_manual_download import build_manual_us_station_row, discover_us_manual_station_ids
from .utils import ensure_directory, read_table, write_table


def import_manual_us_subdaily(
    *,
    manual_download_dir: str | Path,
    audit_path: str | Path,
    examples_gpkg_path: str | Path,
    examples_csv_path: str | Path | None = None,
    subdaily_gpkg_path: str | Path | None = None,
    output_dir: str | Path,
    gauges_cleaned_path: str | Path,
    now_utc: datetime | None = None,
    target_start_date: date = date(2010, 1, 1),
    minimum_completeness: float = 0.70,
    max_gap_days: float = 183.0,
) -> dict[str, Any]:
    manual_download_dir = Path(manual_download_dir)
    audit_path = Path(audit_path)
    examples_gpkg_path = Path(examples_gpkg_path)
    examples_csv_path = None if examples_csv_path is None else Path(examples_csv_path)
    subdaily_gpkg_path = None if subdaily_gpkg_path is None else Path(subdaily_gpkg_path)
    output_dir = Path(output_dir)
    gauges_cleaned_path = Path(gauges_cleaned_path)
    runtime_now = now_utc or datetime.now(timezone.utc)

    station_ids = discover_us_manual_station_ids(manual_download_dir)
    if not station_ids:
        raise ValueError(f"No manual US ZIP downloads were found in {manual_download_dir}.")

    audit = pd.read_csv(audit_path)
    examples = gpd.read_file(examples_gpkg_path, layer="hierarchy_examples_filtered")
    existing_station_keys = set(audit.get("station_key", pd.Series(dtype="string")).astype(str))

    manual_rows: list[dict[str, Any]] = []
    for station_id in station_ids:
        station_key = f"US:{station_id}"
        if station_key in existing_station_keys:
            row = audit[audit["station_key"].astype(str) == station_key].iloc[0].to_dict()
            if station_key not in set(examples["station_key"].astype(str)):
                example_row = build_manual_us_station_row(
                    station_id,
                    examples=examples,
                    gauges_cleaned_path=gauges_cleaned_path,
                )
                row["lat"] = example_row["lat"]
                row["lon"] = example_row["lon"]
                row["example_id"] = example_row["example_id"]
                row["down"] = example_row["down"]
            manual_rows.append(row)
            continue

        row = build_manual_us_station_row(
            station_id,
            examples=examples,
            gauges_cleaned_path=gauges_cleaned_path,
        )
        manual_rows.append(row)

    if not manual_rows:
        raise ValueError("No manual US stations were prepared for import.")

    audit_updated = _merge_manual_rows_into_audit(audit, manual_rows)
    write_table(audit_updated, audit_path)

    examples_updated = _merge_manual_rows_into_examples_gpkg(examples, manual_rows)
    _write_single_layer_gpkg(examples_updated, examples_gpkg_path, layer="hierarchy_examples_filtered")

    if examples_csv_path is not None and examples_csv_path.exists():
        examples_csv = pd.read_csv(examples_csv_path)
        examples_csv_updated = _merge_manual_rows_into_examples_csv(examples_csv, manual_rows)
        write_table(examples_csv_updated, examples_csv_path)

    subset_audit = audit_updated[audit_updated["station_key"].astype(str).isin({row["station_key"] for row in manual_rows})].copy()
    with tempfile.NamedTemporaryFile(prefix="manual_us_subdaily_", suffix=".csv", delete=False) as handle:
        subset_path = Path(handle.name)
    subset_audit.to_csv(subset_path, index=False)
    try:
        timeseries_new, manifest_new = download_subdaily_from_audit(
            subset_path,
            countries=["US"],
            target_start_date=target_start_date,
            minimum_completeness=minimum_completeness,
            max_gap_days=max_gap_days,
            provider_contexts={"usgs": {"manual_download_dir": manual_download_dir}},
            now_utc=runtime_now,
        )
    finally:
        subset_path.unlink(missing_ok=True)

    us_dir = ensure_directory(output_dir / "US")
    timeseries_path = us_dir / "subdaily_timeseries.parquet"
    manifest_path = us_dir / "subdaily_download_manifest.csv"
    summary_path = output_dir / "subdaily_country_download_summary.csv"

    timeseries_existing = read_table(timeseries_path) if timeseries_path.exists() else pd.DataFrame(columns=timeseries_new.columns)
    manifest_existing = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(columns=manifest_new.columns)

    imported_keys = set(manifest_new["station_key"].astype(str))
    timeseries_merged = pd.concat(
        [
            timeseries_existing[~timeseries_existing["station_key"].astype(str).isin(imported_keys)].copy(),
            timeseries_new,
        ],
        ignore_index=True,
        sort=False,
    )
    timeseries_merged = timeseries_merged.sort_values(["station_key", "time"]).reset_index(drop=True)
    write_table(timeseries_merged, timeseries_path)

    manifest_merged = pd.concat(
        [
            manifest_existing[~manifest_existing["station_key"].astype(str).isin(imported_keys)].copy(),
            manifest_new,
        ],
        ignore_index=True,
        sort=False,
    )
    manifest_merged = manifest_merged.sort_values(["country", "station_key"]).reset_index(drop=True)
    write_table(manifest_merged, manifest_path)

    _update_country_summary(
        summary_path=summary_path,
        country="US",
        manifest=manifest_merged,
        timeseries_path=timeseries_path,
        manifest_path=manifest_path,
        requested_start=target_start_date,
        requested_end=runtime_now,
        timeseries_row_count=len(timeseries_merged),
    )

    if subdaily_gpkg_path is not None:
        export_subdaily_hierarchy_package(
            examples_gpkg_path,
            audit_path,
            subdaily_gpkg_path,
            manifests_dir=output_dir,
        )

    return {
        "station_ids": station_ids,
        "imported_station_keys": sorted(imported_keys),
        "audit_rows": len(audit_updated),
        "examples_rows": len(examples_updated),
        "us_manifest_rows": len(manifest_merged),
        "us_timeseries_rows": len(timeseries_merged),
    }


def _merge_manual_rows_into_audit(audit: pd.DataFrame, manual_rows: list[dict[str, Any]]) -> pd.DataFrame:
    prepared_rows = []
    for row in manual_rows:
        prepared = {column: pd.NA for column in audit.columns}
        prepared.update({key: value for key, value in row.items() if key in prepared})
        prepared_rows.append(prepared)
    manual_frame = pd.DataFrame.from_records(prepared_rows, columns=audit.columns)
    combined = pd.concat(
        [
            audit[~audit["station_key"].astype(str).isin(manual_frame["station_key"].astype(str))].copy(),
            manual_frame,
        ],
        ignore_index=True,
    )
    return combined.sort_values(["country", "station_key"]).reset_index(drop=True)


def _merge_manual_rows_into_examples_gpkg(
    examples: gpd.GeoDataFrame,
    manual_rows: list[dict[str, Any]],
) -> gpd.GeoDataFrame:
    existing_keys = set(examples["station_key"].astype(str))
    additions: list[dict[str, Any]] = []
    for row in manual_rows:
        if row["station_key"] in existing_keys:
            continue
        lat = float(row["lat"])
        lon = float(row["lon"])
        additions.append(
            {
                "station_key": row["station_key"],
                "lat": lat,
                "lon": lon,
                "down": row.get("down", True),
                "example_id": float(row["example_id"]),
                "geometry": Point(lon, lat),
            }
        )
    if not additions:
        return examples
    additions_gdf = gpd.GeoDataFrame(additions, geometry="geometry", crs=examples.crs)
    merged = pd.concat([examples, additions_gdf], ignore_index=True, sort=False)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=examples.crs)


def _merge_manual_rows_into_examples_csv(
    examples_csv: pd.DataFrame,
    manual_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    existing_keys = set()
    for column in ["station_key_up", "station_key_dn", "station_key_side_channel"]:
        if column in examples_csv.columns:
            existing_keys.update(examples_csv[column].dropna().astype(str).tolist())

    rows_to_add: list[dict[str, Any]] = []
    next_index = int(pd.to_numeric(examples_csv.get("Unnamed: 0"), errors="coerce").max() or 0) + 1
    for row in manual_rows:
        station_key = str(row["station_key"])
        if station_key in existing_keys:
            continue
        example_rows = examples_csv[examples_csv["example_id"] == int(row["example_id"])]
        template = example_rows.iloc[0].to_dict() if not example_rows.empty else {column: pd.NA for column in examples_csv.columns}
        template["Unnamed: 0"] = next_index
        next_index += 1
        template["reach_id_dn"] = pd.NA
        template["station_key_dn"] = station_key
        template["lat_dn"] = row["lat"]
        template["lon_dn"] = row["lon"]
        if "reach_id_side_channel" in template:
            template["reach_id_side_channel"] = pd.NA
        if "station_key_side_channel" in template:
            template["station_key_side_channel"] = pd.NA
        if "lat" in template:
            template["lat"] = pd.NA
        if "lon" in template:
            template["lon"] = pd.NA
        rows_to_add.append(template)

    if not rows_to_add:
        return examples_csv
    combined = pd.concat([examples_csv, pd.DataFrame.from_records(rows_to_add)], ignore_index=True, sort=False)
    return combined


def _write_single_layer_gpkg(frame: gpd.GeoDataFrame, path: Path, *, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frame.to_file(path, layer=layer, driver="GPKG")


def _update_country_summary(
    *,
    summary_path: Path,
    country: str,
    manifest: pd.DataFrame,
    timeseries_path: Path,
    manifest_path: Path,
    requested_start: date,
    requested_end: datetime,
    timeseries_row_count: int,
) -> None:
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    status_counts = manifest["download_status"].value_counts(dropna=False).to_dict() if not manifest.empty else {}
    row = pd.DataFrame(
        [
            {
                "country": country,
                "requested_start": f"{requested_start.isoformat()}T00:00:00Z",
                "requested_end": requested_end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "station_count": int(len(manifest)),
                "ok_count": int(status_counts.get("ok", 0)),
                "error_count": int(status_counts.get("error", 0)),
                "no_data_count": int(status_counts.get("no_data_returned", 0)),
                "missing_provider_station_id_count": int(status_counts.get("missing_provider_station_id", 0)),
                "timeseries_row_count": int(timeseries_row_count),
                "timeseries_path": str(timeseries_path),
                "manifest_path": str(manifest_path),
            }
        ]
    )
    if not summary.empty and "country" in summary.columns:
        summary["country"] = summary["country"].astype("string").str.upper()
        summary = summary[summary["country"] != country].copy()
        summary = pd.concat([summary, row], ignore_index=True, sort=False)
    else:
        summary = row
    summary["country"] = summary["country"].astype("string").str.upper()
    summary = summary.sort_values("country").reset_index(drop=True)
    write_table(summary, summary_path)
