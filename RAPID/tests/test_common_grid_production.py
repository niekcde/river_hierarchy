import numpy as np
import pandas as pd
import pytest

from rapid_tools import cunge
from rapid_tools.common_grid_production import refined_parameters, shared_clock, model_feature, sparse_parity


def grid():
    return cunge.freeze_grid(pd.DataFrame(dict(id_link=[1, 2, 3, 4], u=[0, 1, 1, 2], v=[1, 2, 2, 3],
        link_length_m=[100., 500., 300., 700.], width=[10., 6., 4., 10.],
        slope=[.001]*4, depth_m=[2.]*4)), 20.)


@pytest.mark.parametrize('beta,multiplier', [(1., 1), (.1, 4), (.0475, 8), (1.5, 1)])
def test_inherited_diffusion_and_grid_preserve_physical_lengths(beta, multiplier):
    original = grid()
    f = refined_parameters(original, dict(n_manning=.035, beta=beta, subdivision_multiplier=multiplier))
    np.testing.assert_array_equal(f['count'], original['count']*multiplier)
    np.testing.assert_allclose(f.rapid_effective_length_m*f['count'], original.link_length_m)
    np.testing.assert_allclose(f.rapid_celerity_mps*f.rapid_effective_length_m*(.5-f.rapid_x),
                               beta*f.hydraulic_diffusivity_m2_s)
    dt = shared_clock([f], [30, 60], 15)
    assert 30/dt == pytest.approx(round(30/dt))
    assert dt < (2*f.rapid_k*(1-f.rapid_x)).min()
    parity = sparse_parity(f, np.array([2., 3., 5., 9., 8., 4., 2.]), 60, dt, 2.)
    assert parity['max_difference_cms'] < 1e-8
    assert parity['output_averaging'] == 'linear_state_interval_mean'


def test_invalid_beta_or_subdivision_cannot_enter_production():
    for beta in [0, -1, np.nan]:
        with pytest.raises(ValueError):
            cunge.parameters(grid(), .035, beta=beta)
    with pytest.raises(ValueError):
        cunge.parameters(grid(), .035, method='fixed_x', beta=.2)
    with pytest.raises(ValueError):
        refined_parameters(grid(), dict(n_manning=.035, beta=1., subdivision_multiplier=1.5))


def test_two_lobe_feature_uses_same_physical_separation_at_both_resolutions():
    review = dict(crest_start_utc='2020-01-01T00:00Z', crest_end_utc='2020-01-01T00:30Z')
    results = []
    for step in [60, 30]:
        t = np.arange(0, 3601, step)
        q = 2+np.exp(-((t-1200)/150)**2)+np.exp(-((t-2400)/150)**2)
        results.append(model_feature(q, t, '54', review))
    assert results[0] == results[1]
    assert results[0][0] == 1800
    assert results[0][2] == 'two_lobe_midpoint'


def test_plateau_feature_retains_midpoint():
    time, height, kind = model_feature([1., 4., 4., 4., 2.], [30, 90, 150, 210, 270], '03', {})
    assert (time, height, kind) == (150., 4., 'exact_plateau_midpoint')


def test_production_forcing_rejects_unapproved_gap_and_marks_explicit_diagnostic(tmp_path):
    from rapid_tools.common_grid_production import CalibrationCase
    case = CalibrationCase.__new__(CalibrationCase)
    case.folder = tmp_path
    case.review = dict(upstream_sampling_minutes=1, downstream_sampling_minutes=1)
    start = pd.Timestamp('2020-01-01', tz='UTC')
    case.wave = pd.DataFrame(dict(interval_start_utc=[start], interval_end_utc=[start+pd.Timedelta(seconds=180)]))
    points = pd.DataFrame(dict(time=[start, start+pd.Timedelta(seconds=180)], discharge_cms=[2., 8.]))
    for side in ['upstream', 'downstream']:
        points.to_parquet(tmp_path/f'observed_{side}.parquet')
    with pytest.raises(ValueError, match='gap'):
        case.forcing(30)
    result = case.forcing(30, diagnostic_gaps=True)
    assert result.upstream_gap.all() and result.downstream_gap.all()
    assert not result.downstream_supported.any()
    assert (result.forcing_cms*30).sum() == pytest.approx(900.)
    assert result.time_utc.iloc[0] == start+pd.Timedelta(seconds=15)
