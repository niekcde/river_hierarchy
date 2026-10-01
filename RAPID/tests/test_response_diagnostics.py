from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapid_tools.response_diagnostics import (
    fit_response_relationships, make_signed_magnitude_scatter,
    response_scores, speed_change_pct,
)


def test_log_magnitude_does_not_hide_sign_errors_or_zero_exclusions():
    scores = response_scores([0, -1, -10, -100], [0, 1, 10, 100])
    assert scores["r2_log10_magnitude"] == pytest.approx(1)
    assert scores["r2_linear"] < 0
    assert scores["r2_signed_log1p"] < 0
    assert scores["sign_accuracy"] == 0.25
    assert scores["n_log_magnitude"] == 3
    assert scores["n_zero_pairs_excluded"] == 1


def test_speed_is_inverse_travel_time_not_negative_lag_change():
    speed = speed_change_pct(pd.Series([-20, 25, 0, -100, -120, np.inf]))
    np.testing.assert_allclose(speed[:3], [25, -20, 0])
    assert speed[3:].isna().all()


def test_plots_retain_every_sign_and_display_zero_without_mutation():
    frame = pd.DataFrame({"x": [2., -3., 0., 4.], "y": [-4., 5., 0., 2.],
                          "plot_category": ["nested unit"] * 4})
    original = frame.copy(deep=True)
    fig = make_signed_magnitude_scatter(frame, x="x", y="y", log_x=True)
    pd.testing.assert_frame_equal(frame, original)
    assert sum(len(t.x) for t in fig.data) == 4
    assert all(np.all(np.asarray(t.x) > 0) and np.all(np.asarray(t.y) > 0) for t in fig.data)
    traces = {t.meta["sign"]: t for t in fig.data}
    assert traces["x +, lag −"].marker.symbol == "circle"
    assert traces["x −, lag +"].marker.symbol == "diamond"
    assert list(traces["x −, lag +"].x) == [3]
    assert list(traces["x 0, lag 0"].customdata[0]) == [0, 0]
    assert fig.layout.xaxis.ticktext[0] == "0 (display)"


def test_grouped_curves_recover_known_power_and_exclude_zeros():
    u = np.tile(np.geomspace(0.01, 10, 12), 4)
    frame = pd.DataFrame({"x": u, "y": -3*u**1.7, "example_id": np.repeat(range(4), 12)})
    frame.loc[len(frame)] = [0, 0, 0]
    scores, params, predictions = fit_response_relationships(frame, x="x", y="y")
    power = scores.loc[scores.model.eq("power_law")].iloc[0]
    assert power.n == 48 and power.n_cv == 48
    assert power.r2_log10_cv == pytest.approx(1)
    assert power.r2_linear_cv == pytest.approx(1)
    assert params.loc[params.model.eq("power_law"), "b"].iloc[0] == pytest.approx(1.7)
    assert predictions.observed_magnitude.gt(0).all()


@pytest.mark.parametrize("log_x,log_y", [(False, False), (False, True), (True, True)])
def test_speed_triplet_preserves_signs_and_marks_faster_states(log_x, log_y):
    frame = pd.DataFrame({"x": [2., -3., 4.], "speed": [25., -20., 0.],
                          "Lambda_matched": [0.1, 10., 0.2],
                          "plot_category": ["nested unit"] * 3})
    fig = make_signed_magnitude_scatter(frame, x="x", y="speed", log_x=log_x, log_y=log_y)
    traces = {t.meta["response_sign"]: t for t in fig.data}
    assert fig.layout.yaxis.title.text == "Speed change (%)"
    assert fig.layout.meta["response_name"] == "Speed"
    assert traces["+"].marker.symbol == "circle"
    assert traces["−"].meta["resistance_sign"] == "−"
    assert list(traces["−"].y) == ([20.] if log_y else [-20.])
    assert list(traces["−"].x) == ([3.] if log_x else [-3.])
    assert (traces["0"].y[0] > 0) if log_y else (traces["0"].y[0] == 0)
    if not log_y:
        assert all(t.marker.symbol == "circle" for t in fig.data)


def test_resistance_sign_uses_log_ratio_even_when_exposure_is_zero():
    frame = pd.DataFrame({"x": [0., 0., 0.], "y": [-1., -1., -1.],
                          "Lambda_matched": [0.1, 1., 10.],
                          "plot_category": ["nested unit"] * 3})
    fig = make_signed_magnitude_scatter(frame, x="x", y="y")
    assert {t.meta["resistance_sign"] for t in fig.data} == {"+", "−", "0"}


def test_single_network_has_no_held_out_score():
    frame = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [-1, -4, -9, -16, -25], "example_id": "one"})
    scores, _, _ = fit_response_relationships(frame, x="x", y="y")
    assert scores.n_cv.eq(0).all()
    assert scores.r2_log10_cv.isna().all()
