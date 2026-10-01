from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


RAPID_ROOT = Path(__file__).resolve().parents[1]
if str(RAPID_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT / "src"))

from rapid_tools.reference_profile import (  # noqa: E402
    ReferenceProfileConfig,
    apply_manual_linear_profile_bridge,
    fit_reference_profile,
    select_reference_profile,
    write_reference_profile_artifacts,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = np.arange(0.0, 1_100.0, 100.0)
    static = pd.DataFrame(
        {
            "reference_section_dist_out_m": x,
            "static_wse_m": 10.0 + 0.01 * x,
        }
    )
    observations = []
    for acquisition, offset in (("2025-01-01", 0.0), ("2025-01-02", 1.2), ("2025-01-03", -0.7)):
        for distance in x:
            observations.append(
                {
                    "reference_section_dist_out_m": distance,
                    "acquisition_id": acquisition,
                    "time": acquisition,
                    "wse_m": 10.0 + 0.01 * distance + offset,
                    "wse_uncertainty_m": 0.2,
                }
            )
    return static, pd.DataFrame(observations)


def test_fit_reference_profile_enforces_gradient_and_centres_offsets() -> None:
    static, observations = _inputs()
    fit = fit_reference_profile(
        static,
        observations,
        endpoint_distances_m=(0.0, 1_000.0),
        lambda_value=100.0,
        config=ReferenceProfileConfig(max_outer_iterations=50, max_inner_iterations=100),
    )

    slopes = fit.profile["interval_slope_m_per_m"].dropna().to_numpy(float)
    assert (slopes >= fit.summary["minimum_slope_m_per_m"] - 1e-12).all()
    assert fit.summary["fitted_global_slope_m_per_m"] == pytest.approx(0.01, abs=1e-5)
    assert fit.acquisition_offsets["stage_offset_m"].mean() == pytest.approx(0.0, abs=1e-12)
    assert fit.observations["residual_m"].abs().max() < 1e-5


def test_select_and_write_reference_profile_artifacts(tmp_path: Path) -> None:
    static, observations = _inputs()
    config = ReferenceProfileConfig(max_outer_iterations=50, max_inner_iterations=100)
    fit, cv, selection = select_reference_profile(
        static,
        observations,
        endpoint_distances_m=(0.0, 1_000.0),
        lambda_values=(1.0, 10.0),
        config=config,
    )
    assert len(cv) == 2
    assert selection["selection_rule"] == "smoothest_within_one_standard_error"

    paths = write_reference_profile_artifacts(
        fit, cv, selection, tmp_path / "profile", config=config,
        input_metadata={"example_id": "synthetic"},
    )
    assert all(path.exists() for path in paths.values())
    assert len(pd.read_parquet(paths["profile"])) == len(fit.profile)
    payload = json.loads(paths["selection"].read_text())
    assert payload["inputs"]["example_id"] == "synthetic"
    assert payload["profile_sha256"]


def test_manual_linear_profile_bridge_is_continuous_and_audited() -> None:
    static, observations = _inputs()
    fit = fit_reference_profile(
        static,
        observations,
        endpoint_distances_m=(0.0, 1_000.0),
        lambda_value=100.0,
        config=ReferenceProfileConfig(max_outer_iterations=50, max_inner_iterations=100),
    )
    bridged, record = apply_manual_linear_profile_bridge(
        fit,
        adjustment_id="synthetic_bridge",
        start_distance_m=300.0,
        end_distance_m=700.0,
    )
    profile = bridged.profile
    bridge = profile.loc[profile["manual_profile_adjustment_active"]]
    assert record["adjustment_type"] == "manual_linear_profile_bridge"
    assert record["n_profile_rows_replaced"] == 5
    assert bridge["manual_profile_adjustment_id"].eq("synthetic_bridge").all()
    assert np.diff(bridge["fitted_wse_m"].to_numpy(float)).std() == pytest.approx(0.0, abs=1e-12)
    assert (np.diff(profile["fitted_wse_m"].to_numpy(float)) > 0.0).all()
