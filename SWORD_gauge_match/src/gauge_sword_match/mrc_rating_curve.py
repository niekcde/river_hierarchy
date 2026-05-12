from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MRC_RATING_CURVE_REFERENCE: dict[str, str] = {
    "title": "Technical Note – Development and update of water level and discharge rating curves for the Mekong mainstream",
    "publisher": "MRC Secretariat",
    "publication_date": "2024-02-20",
    "doi": "10.52107/mrc.bjv4xx",
    "publication_url": "https://www.mrcmekong.org/publications/development-and-update-of-water-level-and-discharge-rating-curves-for-the-mekong-mainstream/",
    "pdf_url": "https://www.mrcmekong.org/wp-content/uploads/2024/09/Technical-Report-Mekong-Rating-Curves.pdf",
    "equation_table_reference": "Table 3 (PDF page 16 / report page 10)",
}


@dataclass(frozen=True, slots=True)
class MrcRatingCurveDefinition:
    discharge_station_id: str
    discharge_station_name: str
    country: str
    coefficient_a: float
    zero_flow_height_m: float
    exponent_m: float
    min_water_level_m: float
    max_water_level_m: float
    min_discharge_cms: float
    max_discharge_cms: float
    stage_station_id: str | None = None
    stage_station_name: str | None = None
    stage_station_relationship: str = "same_station"
    notes: str | None = None

    @property
    def effective_stage_station_id(self) -> str:
        return self.stage_station_id or self.discharge_station_id

    @property
    def effective_stage_station_name(self) -> str:
        return self.stage_station_name or self.discharge_station_name

    @property
    def equation_text(self) -> str:
        sign = "-" if self.zero_flow_height_m >= 0 else "+"
        abs_h0 = abs(self.zero_flow_height_m)
        return f"Q = {self.coefficient_a:.3f} * (H {sign} {abs_h0:.3f}) ^ {self.exponent_m:.3f}"

    def discharge_from_stage(self, water_level_m: pd.Series) -> pd.Series:
        stage = pd.to_numeric(water_level_m, errors="coerce")
        in_range = stage.between(self.min_water_level_m, self.max_water_level_m, inclusive="both")
        base = stage - float(self.zero_flow_height_m)
        discharge = pd.Series(float("nan"), index=stage.index, dtype="float64")
        valid = in_range & base.gt(0.0)
        discharge.loc[valid] = float(self.coefficient_a) * (base.loc[valid] ** float(self.exponent_m))
        return discharge


MRC_RATING_CURVES: dict[str, MrcRatingCurveDefinition] = {
    "011903": MrcRatingCurveDefinition(
        discharge_station_id="011903",
        discharge_station_name="Chiang Khan",
        country="TH",
        coefficient_a=17.587,
        zero_flow_height_m=-2.550,
        exponent_m=2.380,
        min_water_level_m=2.20,
        max_water_level_m=14.30,
        min_discharge_cms=832.0,
        max_discharge_cms=14608.0,
    ),
    "012001": MrcRatingCurveDefinition(
        discharge_station_id="012001",
        discharge_station_name="Nong Khai",
        country="TH",
        coefficient_a=104.725,
        zero_flow_height_m=-3.520,
        exponent_m=1.822,
        min_water_level_m=0.34,
        max_water_level_m=12.36,
        min_discharge_cms=692.0,
        max_discharge_cms=16522.0,
        stage_station_id="012002",
        stage_station_name="Nong Khai 2",
        stage_station_relationship="nearby_stage_proxy",
        notes=(
            "The local MRC telemetry archive is available as station 012002 (Nong Khai 2), while the official "
            "rating equation in the 2024 technical note is for discharge station 012001 (Nong Khai). "
            "This workflow uses 012002 as an explicit nearby stage proxy and records that provenance in the manifest."
        ),
    ),
    "013901": MrcRatingCurveDefinition(
        discharge_station_id="013901",
        discharge_station_name="Pakse",
        country="LA",
        coefficient_a=322.132,
        zero_flow_height_m=-2.220,
        exponent_m=1.808,
        min_water_level_m=0.13,
        max_water_level_m=12.23,
        min_discharge_cms=605.0,
        max_discharge_cms=44565.0,
    ),
    "014501": MrcRatingCurveDefinition(
        discharge_station_id="014501",
        discharge_station_name="Stung Treng",
        country="KH",
        coefficient_a=2149.387,
        zero_flow_height_m=1.220,
        exponent_m=1.365,
        min_water_level_m=1.82,
        max_water_level_m=12.01,
        min_discharge_cms=1640.0,
        max_discharge_cms=57349.0,
    ),
    "014901": MrcRatingCurveDefinition(
        discharge_station_id="014901",
        discharge_station_name="Kratie",
        country="KH",
        coefficient_a=249.180,
        zero_flow_height_m=2.820,
        exponent_m=1.771,
        min_water_level_m=5.96,
        max_water_level_m=22.77,
        min_discharge_cms=1438.0,
        max_discharge_cms=55467.0,
    ),
}


def load_mrc_manual_waterlevel_archive(
    provider_station_id: str,
    waterlevel_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], str] | None:
    normalized_station_id = str(provider_station_id).strip()
    curve = MRC_RATING_CURVES.get(normalized_station_id)
    if curve is None:
        return None

    waterlevel_dir = Path(waterlevel_dir)
    if not waterlevel_dir.exists() or not waterlevel_dir.is_dir():
        return None

    stage_station_id = curve.effective_stage_station_id
    csv_path = _find_station_file(waterlevel_dir, stage_station_id, suffix=".csv")
    if csv_path is None:
        return None
    readme_path = _find_station_file(waterlevel_dir, stage_station_id, suffix="__README.txt")
    readme_metadata = _read_readme_metadata(readme_path)

    frame = pd.read_csv(csv_path)
    if "Timestamp (UTC+07:00)" not in frame.columns or "Value" not in frame.columns:
        raise ValueError(f"MRC water-level archive has an unexpected layout: {csv_path}")

    frame["time"] = pd.to_datetime(frame["Timestamp (UTC+07:00)"], utc=True, errors="coerce")
    frame["water_level_m"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = frame.dropna(subset=["time", "water_level_m"]).copy()
    frame["discharge"] = curve.discharge_from_stage(frame["water_level_m"])

    total_rows = int(len(frame))
    excluded_rows = int(frame["discharge"].isna().sum())
    frame = frame.dropna(subset=["discharge"]).copy()
    if frame.empty:
        return None

    provider_series_id = (
        f"mrc_rating_curve:{curve.discharge_station_id}:stage:{stage_station_id}:{MRC_RATING_CURVE_REFERENCE['doi']}"
    )
    normalized = pd.DataFrame(
        {
            "time": frame["time"],
            "discharge": frame["discharge"],
            "raw_discharge": frame["discharge"],
            "unit_of_measure": "m3/s",
            "raw_unit_of_measure": "m3/s",
            "unit_normalized": True,
            "provider_series_name": "mrc_telemetry_water_level_rating_curve",
            "provider_series_id": provider_series_id,
        }
    )

    manifest_metadata: dict[str, Any] = {
        "derivation_method": "rating_curve_from_telemetry_water_level",
        "waterlevel_station_id": stage_station_id,
        "waterlevel_station_name": curve.effective_stage_station_name,
        "waterlevel_station_relationship": curve.stage_station_relationship,
        "waterlevel_source_csv": str(csv_path),
        "waterlevel_source_readme": str(readme_path) if readme_path is not None else pd.NA,
        "waterlevel_source_location_name": readme_metadata.get("location_name", pd.NA),
        "rating_curve_station_id": curve.discharge_station_id,
        "rating_curve_station_name": curve.discharge_station_name,
        "rating_curve_equation": curve.equation_text,
        "rating_curve_min_water_level_m": float(curve.min_water_level_m),
        "rating_curve_max_water_level_m": float(curve.max_water_level_m),
        "rating_curve_min_discharge_cms": float(curve.min_discharge_cms),
        "rating_curve_max_discharge_cms": float(curve.max_discharge_cms),
        "rating_curve_reference_title": MRC_RATING_CURVE_REFERENCE["title"],
        "rating_curve_reference_doi": MRC_RATING_CURVE_REFERENCE["doi"],
        "rating_curve_reference_url": MRC_RATING_CURVE_REFERENCE["publication_url"],
        "rating_curve_reference_pdf_url": MRC_RATING_CURVE_REFERENCE["pdf_url"],
        "rating_curve_reference_table": MRC_RATING_CURVE_REFERENCE["equation_table_reference"],
        "rating_curve_proxy_applied": bool(stage_station_id != curve.discharge_station_id),
        "rating_curve_out_of_range_row_count": excluded_rows,
        "rating_curve_raw_stage_row_count": total_rows,
    }
    if curve.notes:
        manifest_metadata["rating_curve_station_note"] = curve.notes

    notes = (
        f"MRC telemetry water level from `{csv_path.name}` was converted to discharge using the official MRC 2024 "
        f"mainstream rating curve for station `{curve.discharge_station_id}` "
        f"({curve.equation_text}; {MRC_RATING_CURVE_REFERENCE['equation_table_reference']}; "
        f"DOI {MRC_RATING_CURVE_REFERENCE['doi']})."
    )
    if excluded_rows:
        notes += (
            f" {excluded_rows} water-level rows outside the documented applicability range "
            f"[{curve.min_water_level_m:.2f}, {curve.max_water_level_m:.2f}] m were excluded."
        )
    if curve.notes:
        notes += f" {curve.notes}"

    return normalized, manifest_metadata, notes


def _find_station_file(directory: Path, station_id: str, *, suffix: str) -> Path | None:
    matches = sorted(directory.glob(f"*_{station_id}_*{suffix}"))
    return matches[0] if matches else None


def _read_readme_metadata(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("# "):
                continue
            body = line[2:]
            if ":" not in body:
                continue
            key, value = body.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_").replace("-", "_")
            metadata[normalized_key] = value.strip()
    return metadata
