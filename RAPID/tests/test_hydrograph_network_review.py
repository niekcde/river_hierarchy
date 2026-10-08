import pandas as pd
import pytest
from rapid_tools.hydrograph_review import select_network_pairs


def candidates():
    return pd.DataFrame([
        dict(example_id='05',mode='indep',state_id='S001',technical_alias=False),
        dict(example_id='05',mode='island',state_id='S001',technical_alias=False),
        dict(example_id='05',mode='indep',state_id='alias',technical_alias=True),
        dict(example_id='02',mode='indep',state_id='S001',technical_alias=False),
    ])


def test_all_selection_reports_omitted_and_excludes_aliases():
    selected,omitted=select_network_pairs(candidates(),5,max_pairs=1)
    assert selected['mode'].tolist()==['indep']
    assert omitted['mode'].tolist()==['island']


def test_explicit_mode_disambiguates_state_and_empty_means_none():
    selected,omitted=select_network_pairs(candidates(),'05',units=[('island','S001')])
    assert selected['mode'].tolist()==['island'] and omitted.empty
    assert select_network_pairs(candidates(),'05',units=[])[0].empty


def test_bad_selection_cannot_silently_fall_back_to_all():
    with pytest.raises(ValueError,match='Unknown units'):
        select_network_pairs(candidates(),'05',units=[('indep','missing')])
    with pytest.raises(ValueError,match='positive integer'):
        select_network_pairs(candidates(),'05',max_pairs=0)


@pytest.mark.parametrize('units', [['S001'], ['AB'], [('indep',)], [None]])
def test_selection_requires_mode_state_pairs_even_with_mode_filter(units):
    with pytest.raises(ValueError, match='State IDs alone are not valid'):
        select_network_pairs(candidates(), '05', units=units, modes=['indep'])


def test_model_peak_outside_gauge_window_is_not_silently_clipped(tmp_path, monkeypatch):
    from rapid_tools import hydrograph_review as review
    times=pd.date_range('2024-01-01',periods=6,freq='h',tz='UTC')
    quality=pd.DataFrame([dict(Example=5,Period_start=times[0],Period_end=times[3],Up='u',Down='d')])
    monkeypatch.setattr(review,'_load_gauge_quality_sheet',lambda:quality)
    monkeypatch.setattr(review,'_load_station_timeseries_for_observed_lag',lambda _:pd.DataFrame(dict(time=times,discharge_cms=[1,3,2,1,5,1])))
    monkeypatch.setattr(review,'base_directory',lambda _:tmp_path)
    monkeypatch.setattr(review,'event_clock',lambda *_:(pd.Series(times),'Test UTC'))
    (tmp_path/'run').mkdir()
    pd.DataFrame(dict(q_inflow_cms=[1,3,2,1,1,1],q_outlet_cms=[1,1,2,3,4,5])).to_csv(tmp_path/'run/outlet_hydrograph.csv',index=False)
    _,records=review.gauge_overlay('05')
    assert records.iloc[0].peak_first_utc==times[1]
    model=records[records.series.eq('Model parent outlet')].iloc[0]
    assert model.peak_first_utc==times[5]
    assert not model.peak_inside_gauge_window


def test_paired_hydrographs_preserve_outlet_values_and_require_shared_forcing(tmp_path, monkeypatch):
    from rapid_tools import hydrograph_review as review
    parent_dir=tmp_path/'parent'; child_dir=tmp_path/'child'
    (parent_dir/'run').mkdir(parents=True); child_dir.mkdir()
    parent=pd.DataFrame(dict(time_seconds=[0.,3600.,7200.],
                            time_utc=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC'),
                            q_inflow_cms=[1.,4.,1.],q_outlet_cms=[1.,2.,3.]))
    child=parent.copy(); child['q_outlet_cms']=[1.,3.,2.]
    parent.to_csv(parent_dir/'run/outlet_hydrograph.csv',index=False)
    child.to_csv(child_dir/'outlet_hydrograph.csv',index=False)
    monkeypatch.setattr(review,'base_directory',lambda _:parent_dir)
    monkeypatch.setattr(review,'event_clock',lambda _,h:(pd.to_datetime(h.time_utc,utc=True),'Test UTC'))
    record=pd.Series(dict(example_id='05',mode='island',state_id='test',refreshed_run_dir=str(child_dir)))
    fig=review.parent_child_hydrograph_figure(record,show_forcing=False)
    assert len(fig.data)==2
    assert list(fig.data[0].y)==[1.,2.,3.]
    assert list(fig.data[1].y)==[1.,3.,2.]
    assert len(review.parent_child_hydrograph_figure(record).data)==3
    child.loc[1,'q_inflow_cms']=5.
    child.to_csv(child_dir/'outlet_hydrograph.csv',index=False)
    with pytest.raises(AssertionError):review.parent_child_hydrograph_figure(record)


def test_multi_collapse_overlay_keeps_parent_once_and_checks_every_child(tmp_path, monkeypatch):
    from rapid_tools import hydrograph_review as review
    parent_dir=tmp_path/'parent'
    (parent_dir/'run').mkdir(parents=True)
    parent=pd.DataFrame(dict(time_seconds=[0.,3600.,7200.],
        time_utc=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC'),
        q_inflow_cms=[1.,4.,1.],q_outlet_cms=[1.,2.,3.]))
    parent.to_csv(parent_dir/'run/outlet_hydrograph.csv',index=False)
    monkeypatch.setattr(review,'base_directory',lambda _:parent_dir)
    monkeypatch.setattr(review,'event_clock',lambda _,h:(pd.to_datetime(h.time_utc,utc=True),'Test UTC'))
    rows=[]
    for i,values in enumerate([[1.,3.,2.],[1.,2.5,2.5]]):
        folder=tmp_path/f'child{i}';folder.mkdir()
        child=parent.copy();child.q_outlet_cms=values
        child.to_csv(folder/'outlet_hydrograph.csv',index=False)
        rows.append(dict(example_id='05',mode='island',state_id=f'S{i}',refreshed_run_dir=str(folder)))
    records=pd.DataFrame(rows)
    fig=review.collapse_hydrograph_overlay(records)
    assert len(fig.data)==4
    assert [trace.name for trace in fig.data]==['Shared upstream forcing','Parent outlet','island / S0','island / S1']
    assert list(fig.data[-1].y)==[1.,2.5,2.5]
    assert len(review.collapse_hydrograph_overlay(records,show_forcing=False).data)==3
    child.loc[1,'time_seconds']=3700.
    child.to_csv(folder/'outlet_hydrograph.csv',index=False)
    with pytest.raises(AssertionError):review.collapse_hydrograph_overlay(records)
    with pytest.raises(ValueError,match='one network'):review.collapse_hydrograph_overlay(records.assign(example_id=['05','03']))
    with pytest.raises(ValueError,match='at least one'):review.collapse_hydrograph_overlay(records.iloc[:0])
