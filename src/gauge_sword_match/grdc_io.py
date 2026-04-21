from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ensure_directory

GRDC_DEFAULT_SHEET = "station_catalogue"
GRDC_DAILY_YEARS_ALIASES = ["d_yrs", "Unnamed: 12"]
GRDC_NUMERIC_COLUMNS = [
    "grdc_no",
    "wmo_reg",
    "sub_reg",
    "lat",
    "long",
    "lon",
    "area",
    "drainage_area",
    "altitude",
    "d_start",
    "d_end",
    "d_yrs",
    "Unnamed: 12",
    "d_miss",
    "m_start",
    "m_end",
    "m_yrs",
    "m_miss",
    "t_start",
    "t_end",
    "t_yrs",
    "lta_discharge",
    "r_volume_yr",
    "r_vol_yr",
    "r_height_yr",
]
GRDC_MISSING_TOKENS = {
    "n.a.": pd.NA,
    "n/a": pd.NA,
    "na": pd.NA,
    "nan": pd.NA,
    "none": pd.NA,
    "null": pd.NA,
    "-": pd.NA,
    "": pd.NA,
}


def load_grdc_catalog(path: str | Path, sheet_name: str = GRDC_DEFAULT_SHEET) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError as exc:  # pragma: no cover - depends on local optional engine availability.
        raise RuntimeError(
            "Reading the GRDC Excel catalogue requires an Excel engine such as openpyxl. Install the project dependencies first."
        ) from exc


def prepare_grdc_catalog(
    frame: pd.DataFrame,
    *,
    daily_only: bool = True,
    min_daily_years: float = 1.0,
) -> pd.DataFrame:
    working = frame.copy()
    working = working.replace(
        {
            column: {
                value: replacement
                for value, replacement in GRDC_MISSING_TOKENS.items()
            }
            for column in working.columns
        }
    )
    _copy_column(working, "grdc_no", "station_id")
    _copy_column(working, "station", "station_name")
    _copy_column(working, "river", "river_name")
    _copy_column(working, "long", "lon")
    _copy_column(working, "area", "drainage_area")
    _copy_column(working, "country_code", "country")

    if "agency" not in working.columns:
        working["agency"] = "GRDC"
    else:
        working["agency"] = working["agency"].fillna("GRDC")

    if "grdc_no" not in working.columns and "station_id" in working.columns:
        working["grdc_no"] = working["station_id"]

    if "d_yrs" not in working.columns:
        for alias in GRDC_DAILY_YEARS_ALIASES:
            if alias in working.columns:
                working["d_yrs"] = working[alias]
                break

    working["station_id"] = working["station_id"].astype("string").str.strip()
    working["station_name"] = working["station_name"].astype("string").str.strip()
    working["country"] = working["country"].astype("string").str.upper()
    working["agency"] = working["agency"].astype("string")
    working["river_name"] = working["river_name"].astype("string")
    for column in GRDC_NUMERIC_COLUMNS:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    if daily_only:
        daily_available = working["d_start"].notna() | working["d_end"].notna() | working["d_yrs"].fillna(0).gt(0)
        daily_mask = daily_available & (working["d_yrs"].fillna(min_daily_years).ge(min_daily_years))
        working = working[daily_mask].copy()

    return working.reset_index(drop=True)


def build_grdc_request_table(best_matches: pd.DataFrame) -> pd.DataFrame:
    matched = best_matches[best_matches["reach_id"].notna()].copy()
    if matched.empty:
        return matched

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    matched["confidence_rank"] = matched["confidence_class"].map(confidence_order).fillna(99).astype(int)

    columns = [
        "station_name",
        "grdc_no",
        "station_id",
        "country",
        "river_name",
        "lat",
        "lon",
        "drainage_area",
        "d_start",
        "d_end",
        "d_yrs",
        "d_miss",
        "reach_id",
        "sword_region",
        "sword_node_id",
        "confidence_class",
        "distance_m",
        "total_score",
        "score_gap",
        "review_flag",
    ]
    available = [column for column in columns if column in matched.columns]
    request = matched[available].copy()
    request["confidence_rank"] = matched["confidence_rank"]
    request = request.sort_values(
        ["confidence_rank", "country", "station_name", "grdc_no"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return request.drop(columns=["confidence_rank"])


def write_grdc_request_station_names(request_table: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)

    names = (
        request_table.get("station_name", pd.Series(dtype="string"))
        .dropna()
        .astype("string")
        .str.strip()
    )
    names = names[names.ne("")]
    path.write_text("\n".join(names.tolist()), encoding="utf-8")
    return path


def _copy_column(frame: pd.DataFrame, source: str, target: str) -> None:
    if target not in frame.columns and source in frame.columns:
        frame[target] = frame[source]
