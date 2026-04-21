import pandas as pd

from gauge_sword_match.scoring import distance_score, drainage_area_score, river_name_similarity, score_candidates


def test_distance_score_decays_to_zero_at_radius():
    assert distance_score(0, 5_000) == 1.0
    assert distance_score(2_500, 5_000) == 0.25
    assert distance_score(5_000, 5_000) == 0.0


def test_river_name_similarity_rewards_close_names():
    close_score = river_name_similarity("Rio Grande", "Rio Grande")
    far_score = river_name_similarity("Rio Grande", "Mississippi")

    assert close_score is not None
    assert far_score is not None
    assert close_score > 0.95
    assert far_score < close_score


def test_drainage_area_score_penalizes_large_mismatch():
    assert drainage_area_score(1000, 1000) == 1.0
    assert drainage_area_score(1000, 1000000) < 0.1


def test_score_candidates_handles_all_missing_component_column():
    candidates = pd.DataFrame(
        [
            {
                "station_key": "US:1",
                "distance_m": 100.0,
                "gauge_river_name": None,
                "reach_river_name": None,
                "gauge_drainage_area": None,
                "reach_drainage_proxy": None,
                "reach_id": 1,
            }
        ]
    )

    scored = score_candidates(
        candidates,
        score_weights={"distance": 0.6, "river_name": 0.2, "drainage_area": 0.2},
        search_radius_m=5_000,
        ambiguity_penalty_weight=0.1,
        ambiguity_window=0.05,
    )

    assert len(scored) == 1
    assert scored.loc[0, "total_score"] > 0
