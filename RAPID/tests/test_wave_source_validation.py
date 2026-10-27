"""Wave animations must not substitute archived results for outlet-only runs."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rapid_tools.analysis import _load_wave_state


@pytest.mark.parametrize('qout', [None, '', '   '])
def test_outlet_only_run_rejected_before_artifact_loading(tmp_path, qout):
    pd.DataFrame([dict(state_id='S000_base', method='cunge',
                       output_format='compact_outlet_interval_means', qout_nc=qout)]).to_csv(
        tmp_path / 'rapid_run_registry.csv', index=False)
    with pytest.raises(ValueError, match='no registered reach-time discharge'):
        _load_wave_state(tmp_path, 'S000_base')


def test_missing_cunge_artifact_never_uses_legacy_fallback(tmp_path):
    legacy = tmp_path / 'states/S000_base/rapid/prep'
    legacy.mkdir(parents=True)
    pd.DataFrame([dict(state_id='S000_base', method='cunge',
                       qout_nc=str(tmp_path / 'missing.nc'),
                       rapid_prep_dir=str(tmp_path / 'missing_cunge_prep'))]).to_csv(
        tmp_path / 'rapid_run_registry.csv', index=False)
    with pytest.raises(FileNotFoundError, match='Refusing to substitute legacy'):
        _load_wave_state(tmp_path, 'S000_base')
