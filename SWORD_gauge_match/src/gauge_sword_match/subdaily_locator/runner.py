from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bulgaria import BulgariaAppdClient, locate_bulgaria_subdaily_station
from .brazil import BRAZIL_CURATED_INVENTORY_OVERRIDES, BrazilAnaHydroClient, locate_brazil_subdaily_station
from .canada import CANADA_CURATED_INVENTORY_OVERRIDES, CanadaWaterofficeClient, locate_canada_subdaily_station
from .chile import ChileDgaClient, locate_chile_subdaily_station
from .colombia import ColombiaIdeamFewsClient, locate_colombia_subdaily_station
from .french_guiana import FranceHubeauClient, locate_french_guiana_subdaily_station
from .mekong_mrc import (
    MekongMrcClient,
    locate_cambodia_subdaily_station,
    locate_laos_subdaily_station,
    locate_thailand_subdaily_station,
)
from .niger_basin_abn import NigerBasinAbnClient, locate_mali_subdaily_station
from .nigeria import NigeriaNihsaClient, locate_nigeria_subdaily_station
from .russia import RussiaGmvoClient, locate_russia_subdaily_station
from .inventory import autodetect_inventory_path, enrich_seeds_with_inventory_matches, load_gauge_inventory
from .seeds import load_hierarchy_example_station_seeds
from .usgs import USGSWaterDataClient, locate_usgs_subdaily_station


def locate_subdaily_from_hierarchy_examples(
    input_path: str | Path,
    *,
    country: str,
    layer: str = "hierarchy_examples_filtered",
    search_radius_m: float = 5_000.0,
    nearby_limit: int = 25,
    inventory_path: str | Path | None = None,
    inventory_snap_distance_m: float = 5_000.0,
    max_resolution_distance_m: float = 5_000.0,
    client: USGSWaterDataClient | CanadaWaterofficeClient | BrazilAnaHydroClient | ChileDgaClient | FranceHubeauClient | ColombiaIdeamFewsClient | MekongMrcClient | BulgariaAppdClient | NigerBasinAbnClient | NigeriaNihsaClient | RussiaGmvoClient | None = None,
) -> pd.DataFrame:
    normalized_country = str(country).strip().upper()
    if normalized_country not in {"US", "CA", "BR", "CL", "GF", "CO", "KH", "LA", "TH", "BG", "ML", "NG", "RU"}:
        raise NotImplementedError(
            f"Subdaily locator is currently implemented for US, CA, BR, CL, GF, CO, KH, LA, TH, BG, ML, NG, and RU only; received country '{normalized_country}'."
        )

    seeds = load_hierarchy_example_station_seeds(input_path, layer=layer)
    country_seeds = seeds[seeds["country"] == normalized_country].copy()
    if country_seeds.empty:
        raise ValueError(f"No station seeds found for country '{normalized_country}' in {input_path}")

    resolved_inventory_path = Path(inventory_path) if inventory_path is not None else autodetect_inventory_path(input_path)
    if resolved_inventory_path is not None and resolved_inventory_path.exists():
        inventory = load_gauge_inventory(resolved_inventory_path)
        country_seeds = enrich_seeds_with_inventory_matches(
            country_seeds,
            inventory,
            max_snap_distance_m=inventory_snap_distance_m,
            station_overrides=(
                CANADA_CURATED_INVENTORY_OVERRIDES
                if normalized_country == "CA"
                else (BRAZIL_CURATED_INVENTORY_OVERRIDES if normalized_country == "BR" else None)
            ),
        )

    results: list[dict[str, object]] = []
    for _, row in country_seeds.sort_values("station_key").reset_index(drop=True).iterrows():
        try:
            if normalized_country == "US":
                resolved_client = client or USGSWaterDataClient()
                result = locate_usgs_subdaily_station(
                    row,
                    client=resolved_client,
                    search_radius_m=search_radius_m,
                    nearby_limit=nearby_limit,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "CA":
                resolved_client = client or CanadaWaterofficeClient()
                result = locate_canada_subdaily_station(
                    row,
                    client=resolved_client,
                )
            elif normalized_country == "BR":
                resolved_client = client or BrazilAnaHydroClient()
                result = locate_brazil_subdaily_station(
                    row,
                    client=resolved_client,
                )
            elif normalized_country == "GF":
                resolved_client = client or FranceHubeauClient()
                result = locate_french_guiana_subdaily_station(
                    row,
                    client=resolved_client,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "CO":
                resolved_client = client or ColombiaIdeamFewsClient()
                result = locate_colombia_subdaily_station(
                    row,
                    client=resolved_client,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "KH":
                resolved_client = client or MekongMrcClient()
                result = locate_cambodia_subdaily_station(
                    row,
                    client=resolved_client,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "LA":
                resolved_client = client or MekongMrcClient()
                result = locate_laos_subdaily_station(
                    row,
                    client=resolved_client,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "TH":
                resolved_client = client or MekongMrcClient()
                result = locate_thailand_subdaily_station(
                    row,
                    client=resolved_client,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "BG":
                resolved_client = client or BulgariaAppdClient()
                result = locate_bulgaria_subdaily_station(
                    row,
                    client=resolved_client,
                )
            elif normalized_country == "ML":
                resolved_client = client or NigerBasinAbnClient()
                result = locate_mali_subdaily_station(
                    row,
                    client=resolved_client,
                    max_resolution_distance_m=max_resolution_distance_m,
                )
            elif normalized_country == "NG":
                resolved_client = client or NigeriaNihsaClient()
                result = locate_nigeria_subdaily_station(
                    row,
                    client=resolved_client,
                )
            elif normalized_country == "RU":
                resolved_client = client or RussiaGmvoClient()
                result = locate_russia_subdaily_station(
                    row,
                    client=resolved_client,
                )
            else:
                resolved_client = client or ChileDgaClient()
                result = locate_chile_subdaily_station(
                    row,
                    client=resolved_client,
                )
        except Exception as exc:
            result = {
                "station_key": row["station_key"],
                "country": row["country"],
                "source_station_id": row["source_station_id"],
                "lat": row["lat"],
                "lon": row["lon"],
                "occurrence_count": row["occurrence_count"],
                "example_ids": row["example_ids"],
                "down_values": row["down_values"],
                "provider": (
                    "usgs"
                    if normalized_country == "US"
                    else (
                        "canada_wateroffice"
                        if normalized_country == "CA"
                        else (
                            "brazil_ana"
                            if normalized_country == "BR"
                            else (
                                "france_hubeau"
                                if normalized_country == "GF"
                                else (
                                    "colombia_ideam_fews"
                                    if normalized_country == "CO"
                                    else (
                                        "mrc_timeseries"
                                        if normalized_country in {"KH", "LA", "TH"}
                                        else (
                                            "bulgaria_appd"
                                            if normalized_country == "BG"
                                            else (
                                                "niger_basin_abn"
                                                if normalized_country == "ML"
                                                else ("nigeria_nihsa" if normalized_country == "NG" else ("russia_gmvo" if normalized_country == "RU" else "chile_dga"))
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                ),
                "status": "api_error",
                "resolution_method": "api_error",
                "candidate_site_numbers": "",
                "inventory_station_id": row.get("inventory_station_id"),
                "inventory_station_key": row.get("inventory_station_key"),
                "inventory_station_name": row.get("inventory_station_name"),
                "inventory_resolution_method": row.get("inventory_resolution_method"),
                "inventory_distance_m": row.get("inventory_distance_m"),
                "resolved_monitoring_location_id": None,
                "resolved_site_number": None,
                "resolved_station_name": None,
                "resolution_distance_m": None,
                "monitoring_location_found": False,
                "discharge_series_found": False,
                "subdaily_discharge_found": False,
                "discharge_series_count": 0,
                "instantaneous_series_count": 0,
                "primary_instantaneous_series_count": 0,
                "daily_series_count": 0,
                "instantaneous_begin": None,
                "instantaneous_end": None,
                "daily_begin": None,
                "daily_end": None,
                "daily_coverage_type": None,
                "notes": str(exc),
            }
        results.append(result)

    return pd.DataFrame.from_records(results)
