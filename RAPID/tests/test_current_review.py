"""Protect against a table/plot source mismatch in the current-output review."""
from pathlib import Path
import sys
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rapid_tools.current_review import load_pair, parent_child_hydrograph_figure


@pytest.fixture
def pair(tmp_path):
    t=pd.date_range('2026-04-03',periods=4,freq='min',tz='UTC')
    paths={}
    for label,q in [('parent',[1.,3.,2.,1.]),('child',[1.,2.,4.,1.])]:
        p=tmp_path/f'{label}.parquet'
        pd.DataFrame(dict(time_utc=t,q_outlet_cms=q)).to_parquet(p)
        paths[label]=str(p)
    forcing=tmp_path/'forcing.parquet'
    pd.DataFrame(dict(time_utc=t,forcing_cms=[1.,3.,1.,1.])).to_parquet(forcing)
    return pd.Series(dict(example_id='15',mode='indep',state_id='test',
        parent_outlet_parquet=paths['parent'],outlet_parquet=paths['child'],
        forcing_parquet=str(forcing),parent_forcing_parquet=str(forcing),
        parent_peak_utc=str(t[1]),child_peak_utc=str(t[2]),parent_peak_cms=3.,child_peak_cms=4.,
        sampled_peak_shift_minutes=1.,registry_path='current registry'))


def test_plot_uses_same_series_and_native_utc_as_table(pair):
    fig=parent_child_hydrograph_figure(pair)
    assert list(fig.data[1].y)==[1.,3.,2.,1.]
    assert list(fig.data[2].y)==[1.,2.,4.,1.]
    assert pd.Timestamp(fig.data[2].x[2])==pd.Timestamp(pair.child_peak_utc)
    assert fig.layout.meta['child_outlet']==pair.outlet_parquet


def test_stale_peak_metadata_is_rejected(pair):
    pair.child_peak_utc=pair.parent_peak_utc
    with pytest.raises(ValueError,match='registry maximum timestamp'):
        load_pair(pair)


def test_different_forcing_is_rejected(pair,tmp_path):
    forcing=pd.read_parquet(pair.forcing_parquet);forcing.loc[1,'forcing_cms']=99.
    p=tmp_path/'other.parquet';forcing.to_parquet(p);pair.forcing_parquet=str(p)
    with pytest.raises(AssertionError):load_pair(pair)


def test_island_geometry_uses_recorded_rerun_manifest(tmp_path):
    from rapid_tools.current_review import geometry_manifest_path
    state = tmp_path/'states'/'island_test'
    parameters = state/'rapid'/'cunge'/'prep'/'parameters.parquet'
    context = tmp_path/'island_reruns'/'island_test'/'hydraulic_context'
    context.mkdir(parents=True)
    manifest = context/'hydraulic_context_manifest.json'
    manifest.write_text('{}')
    pd.DataFrame([dict(island_id='island_test', hydraulic_frame_path=str(context/'frame.parquet'))]).to_csv(tmp_path/'island_analysis_manifest.csv', index=False)
    record = pd.Series(dict(state_id='island_test', parameters_parquet=str(parameters), registry_path=str(tmp_path/'rapid_run_registry.csv')))
    assert geometry_manifest_path(record, 'parameters_parquet') == manifest
    record.state_id = 'unknown_island'
    with pytest.raises(FileNotFoundError): geometry_manifest_path(record, 'parameters_parquet')
