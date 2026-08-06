from __future__ import annotations

import pandas as pd
import pytest

from network_variants.width_qc_review_rebase import _changed_dataset_keys


def test_changed_dataset_keys_reports_only_modified_example_links() -> None:
    old = pd.DataFrame(
        {
            "example_id": [2, 2, 15],
            "id_link": [1, 2, 125],
            "manual_decision": ["good", "bad", "good"],
            "feature": [0.1, 0.2, 0.3],
        }
    )
    new = old.copy()
    new.loc[new["id_link"].eq(125), "manual_decision"] = "bad"
    new.loc[new["id_link"].eq(2), "feature"] = 0.25

    assert _changed_dataset_keys(old, new) == [("02", 2), ("15", 125)]


def test_changed_dataset_keys_rejects_membership_change() -> None:
    old = pd.DataFrame({"example_id": ["02"], "id_link": [1], "feature": [0.1]})
    new = pd.DataFrame({"example_id": ["02"], "id_link": [2], "feature": [0.1]})

    with pytest.raises(ValueError, match="membership changes"):
        _changed_dataset_keys(old, new)
