from __future__ import annotations

import math
from difflib import SequenceMatcher

import pandas as pd

from .utils import normalize_text

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback is exercised in environments without rapidfuzz.
    fuzz = None


def distance_score(distance_m: float | None, search_radius_m: float) -> float | None:
    if distance_m is None or pd.isna(distance_m) or search_radius_m <= 0:
        return None
    ratio = min(max(float(distance_m) / float(search_radius_m), 0.0), 1.0)
    return max(0.0, (1.0 - ratio) ** 2)


def river_name_similarity(gauge_name: object, reach_name: object) -> float | None:
    left = normalize_text(gauge_name)
    right = normalize_text(reach_name)
    if not left or not right or left == "nodata" or right == "nodata":
        return None
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left, right)) / 100.0
    return SequenceMatcher(None, left, right).ratio()


def drainage_area_score(gauge_area: object, reach_area: object) -> float | None:
    if gauge_area is None or reach_area is None or pd.isna(gauge_area) or pd.isna(reach_area):
        return None
    try:
        gauge_value = float(gauge_area)
        reach_value = float(reach_area)
    except (TypeError, ValueError):
        return None
    if gauge_value <= 0 or reach_value <= 0:
        return None
    delta = abs(math.log10(gauge_value / reach_value))
    return max(0.0, 1.0 - (delta / 1.5))


def score_candidates(
    candidates: pd.DataFrame,
    score_weights: dict[str, float],
    search_radius_m: float,
    ambiguity_penalty_weight: float,
    ambiguity_window: float,
) -> pd.DataFrame:
    if candidates.empty:
        result = candidates.copy()
        result["distance_score"] = pd.Series(dtype="float64")
        result["river_name_score"] = pd.Series(dtype="float64")
        result["drainage_area_score"] = pd.Series(dtype="float64")
        result["raw_score"] = pd.Series(dtype="float64")
        result["ambiguity_penalty"] = pd.Series(dtype="float64")
        result["total_score"] = pd.Series(dtype="float64")
        return result

    scored = candidates.copy()
    scored["distance_score"] = scored["distance_m"].apply(lambda value: distance_score(value, search_radius_m))
    scored["river_name_score"] = scored.apply(
        lambda row: river_name_similarity(row.get("gauge_river_name"), row.get("reach_river_name")),
        axis=1,
    )
    scored["drainage_area_score"] = scored.apply(
        lambda row: drainage_area_score(row.get("gauge_drainage_area"), row.get("reach_drainage_proxy")),
        axis=1,
    )

    component_columns = {
        "distance": "distance_score",
        "river_name": "river_name_score",
        "drainage_area": "drainage_area_score",
    }
    weighted_sum = pd.Series(0.0, index=scored.index)
    available_weights = pd.Series(0.0, index=scored.index)

    for component_name, column_name in component_columns.items():
        weight = float(score_weights.get(component_name, 0.0))
        if weight <= 0:
            continue
        component_values = pd.to_numeric(scored[column_name], errors="coerce")
        mask = component_values.notna()
        if not bool(mask.any()):
            continue
        weighted_sum = weighted_sum.add(component_values.fillna(0.0) * weight, fill_value=0.0)
        available_weights = available_weights.add(mask.astype(float) * weight, fill_value=0.0)

    scored["raw_score"] = weighted_sum.where(available_weights > 0, 0.0) / available_weights.where(
        available_weights > 0, 1.0
    )

    penalties: list[tuple[int, float]] = []
    for _, group in scored.groupby("station_key"):
        top_raw = float(group["raw_score"].max())
        close_count = int((group["raw_score"] >= top_raw - ambiguity_window).sum())
        penalty = 0.0
        if len(group) > 1 and close_count > 1:
            penalty = ambiguity_penalty_weight * ((close_count - 1) / (len(group) - 1))
        penalties.extend((idx, penalty) for idx in group.index)

    penalty_series = pd.Series(dict(penalties))
    scored["ambiguity_penalty"] = scored.index.to_series().map(penalty_series).fillna(0.0)
    scored["total_score"] = (scored["raw_score"] - scored["ambiguity_penalty"]).clip(lower=0.0)
    scored = scored.sort_values(
        ["station_key", "total_score", "distance_m", "reach_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    return scored
