from __future__ import annotations

import pytest

from RAPID.run_materialize_slope_refit_candidates import (
    REVIEWED_LAMBDA_BY_EXAMPLE,
    REVIEWED_MANUAL_PROFILE_BRIDGES_BY_EXAMPLE,
    _canonical_example_id,
)


def test_reviewed_lambda_registry_has_all_direction_corrected_examples() -> None:
    assert set(REVIEWED_LAMBDA_BY_EXAMPLE) == {
        "02", "03", "05", "11", "13", "15", "21", "22", "37", "41", "45", "46", "48", "54", "55", "56"
    }
    assert REVIEWED_LAMBDA_BY_EXAMPLE["11"] == pytest.approx(3.981072e5)
    assert REVIEWED_LAMBDA_BY_EXAMPLE["41"] == pytest.approx(1.995262e11)


def test_canonical_example_id_validates_population() -> None:
    assert _canonical_example_id("2") == "02"
    with pytest.raises(ValueError, match="not in the reviewed"):
        _canonical_example_id("4")


def test_reviewed_manual_profile_bridge_registry_is_explicit() -> None:
    assert REVIEWED_MANUAL_PROFILE_BRIDGES_BY_EXAMPLE == {
        "46": (("manual_profile_bridge_46_370_410", 369970.16393422784, 410002.52962018247),)
    }
