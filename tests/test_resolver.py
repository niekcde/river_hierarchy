import pandas as pd

from gauge_sword_match.config import MatchingConfig
from gauge_sword_match.resolver import resolve_best_matches


def test_resolve_best_matches_keeps_best_candidate_and_unmatched_rows():
    gauges = pd.DataFrame(
        [
            {
                "station_key": "US:a",
                "station_id": "a",
                "station_name": "Gauge A",
                "lat": 10.0,
                "lon": 20.0,
                "country": "US",
                "agency": "demo",
                "drainage_area": 100.0,
                "river_name": "Alpha",
            },
            {
                "station_key": "US:b",
                "station_id": "b",
                "station_name": "Gauge B",
                "lat": 11.0,
                "lon": 21.0,
                "country": "US",
                "agency": "demo",
                "drainage_area": 100.0,
                "river_name": "Beta",
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {
                "station_key": "US:a",
                "reach_id": 101,
                "sword_region": "na",
                "source_file": "na.parquet",
                "distance_m": 120.0,
                "reach_river_name": "Alpha",
                "reach_drainage_proxy": 110.0,
                "candidate_rank": 1,
                "raw_score": 0.95,
                "ambiguity_penalty": 0.02,
                "total_score": 0.93,
                "distance_score": 0.90,
                "river_name_score": 1.00,
                "drainage_area_score": 0.95,
            },
            {
                "station_key": "US:a",
                "reach_id": 102,
                "sword_region": "na",
                "source_file": "na.parquet",
                "distance_m": 700.0,
                "reach_river_name": "Alpha Branch",
                "reach_drainage_proxy": 60.0,
                "candidate_rank": 2,
                "raw_score": 0.74,
                "ambiguity_penalty": 0.02,
                "total_score": 0.72,
                "distance_score": 0.75,
                "river_name_score": 0.80,
                "drainage_area_score": 0.70,
            },
        ]
    )

    resolved = resolve_best_matches(gauges, scored, MatchingConfig())

    matched = resolved.loc[resolved["station_key"] == "US:a"].iloc[0]
    unmatched = resolved.loc[resolved["station_key"] == "US:b"].iloc[0]

    assert matched["reach_id"] == 101
    assert matched["second_best_score"] == 0.72
    assert round(float(matched["score_gap"]), 2) == 0.21
    assert matched["confidence_class"] == "high"

    assert pd.isna(unmatched["reach_id"])
    assert unmatched["confidence_class"] == "unmatched"
    assert bool(unmatched["review_flag"]) is True
