from pathlib import Path
import sys

import numpy as np
import pytest
from shapely.geometry import LineString, Point, MultiLineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rapid_tools.speed_reference import absolute_speed_change, projected_gauge_length


def test_absolute_speed_uses_seconds_and_example_specific_reference_speed():
    # 7.2 km in 2 h means 1 m/s; a 20% shorter lag means +0.25 m/s.
    result = absolute_speed_change([-20, 25, -20, 0], [7200, 7200, 14400, 7200], 2)
    np.testing.assert_allclose(result, [.25, -.2, .5, 0])


def test_invalid_reference_or_adjusted_time_has_no_speed_estimate():
    result = absolute_speed_change([-100, -110, -20, -20, np.nan], 1000, [2, 2, 0, -1, 2])
    assert np.isnan(result).all()


def test_distance_follows_centreline_between_gauges_not_entire_line():
    line = LineString([(0, 0), (3000, 0), (3000, 4000)])
    length, up_offset, down_offset = projected_gauge_length(line, Point(1000, 100), Point(3100, 3000))
    assert length == 5000
    assert up_offset == down_offset == 100
    assert projected_gauge_length(LineString(list(line.coords)[::-1]), Point(1000, 100), Point(3100, 3000))[0] == length


def test_disconnected_reference_is_rejected():
    with pytest.raises(ValueError, match="continuous"):
        projected_gauge_length(MultiLineString([[(0, 0), (1, 0)], [(3, 0), (4, 0)]]), Point(0, 0), Point(4, 0))
