from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from network_variants.width_samples import read_width_samples, resolve_width_samples_path


def test_read_width_samples_prefers_parquet_over_legacy_csv(tmp_path: Path) -> None:
    pd.DataFrame({"id_link": [1], "width_wet": [8.0]}).to_csv(
        tmp_path / "link_width_samples.csv", index=False
    )
    pd.DataFrame({"id_link": [2], "width_wet": [9.0]}).to_parquet(
        tmp_path / "link_width_samples.parquet", index=False
    )

    assert resolve_width_samples_path(tmp_path).name == "link_width_samples.parquet"
    assert read_width_samples(tmp_path).to_dict("list") == {
        "id_link": [2],
        "width_wet": [9.0],
    }


def test_read_width_samples_accepts_legacy_csv_and_reports_missing(tmp_path: Path) -> None:
    pd.DataFrame({"id_link": [1]}).to_csv(tmp_path / "link_width_samples.csv", index=False)
    assert read_width_samples(tmp_path).to_dict("list") == {"id_link": [1]}
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_width_samples(empty)


def test_read_width_samples_accepts_an_explicit_legacy_filename(tmp_path: Path) -> None:
    source = tmp_path / "parent_samples.csv"
    pd.DataFrame({"id_link": [1], "width_wet": [12.0]}).to_csv(source, index=False)

    assert resolve_width_samples_path(source) == source
    assert read_width_samples(source).to_dict("list") == {
        "id_link": [1],
        "width_wet": [12.0],
    }
