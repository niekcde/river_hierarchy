from __future__ import annotations

from pathlib import Path

import pandas as pd

from network_variants.width_review_model import (
    DEFAULT_FEATURE_COLUMNS,
    build_training_table,
)


def test_build_training_table_discovers_and_combines_reviews(tmp_path: Path) -> None:
    root = tmp_path / "output"
    csv_a = root / "02_indep__anchored" / "states" / "S000_base" / "variant" / "width_review" / "width_review_decisions.csv"
    csv_b = root / "03_indep__anchored" / "states" / "S000_base" / "variant" / "width_review" / "width_review_decisions.csv"
    csv_a.parent.mkdir(parents=True, exist_ok=True)
    csv_b.parent.mkdir(parents=True, exist_ok=True)

    frame_a = pd.DataFrame(
        {
            "id_link": [1, 2],
            "review_decision": ["good", "bad"],
            "qc_mean_opposition_cos_trimmed_from_samples": [-0.9, -0.4],
            "qc_median_opposition_cos_trimmed_from_samples": [-0.9, -0.3],
            "us_signed_dev": [0.1, 0.7],
            "ds_signed_dev": [0.0, 0.5],
            "split_sensitive": [False, True],
            "n_trimmed_qc_samples": [4, 2],
        }
    )
    frame_b = pd.DataFrame(
        {
            "id_link": [3],
            "review_decision": ["bad"],
            "qc_mean_opposition_cos_trimmed_from_samples": [-0.2],
            "qc_median_opposition_cos_trimmed_from_samples": [-0.1],
            "us_signed_dev": [0.8],
            "ds_signed_dev": [0.6],
            "split_sensitive": [True],
            "n_trimmed_qc_samples": [1],
        }
    )
    frame_a.to_csv(csv_a, index=False)
    frame_b.to_csv(csv_b, index=False)

    dataset = build_training_table(root)

    assert len(dataset.review_csv_paths) == 2
    assert len(dataset.frame) == 3
    assert set(dataset.frame["example_id"]) == {"02", "03"}
    for column in DEFAULT_FEATURE_COLUMNS:
        assert column in dataset.frame.columns
    assert set(dataset.frame["target"]) == {0, 1}
