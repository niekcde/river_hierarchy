import numpy as np
import pandas as pd
import pytest
import context_timing_tests as m
from types import SimpleNamespace


def test_parent_onset_and_duration():
    t=np.arange(120)*60.;q=np.r_[np.zeros(20),np.linspace(1,100,50),np.full(50,30.)]
    r=m.parent_reference(t,q,0.,0.,t[69])
    assert r['t0_seconds']==t[20]
    assert r['parent_time_to_peak_hours']==49/60
    assert r['tolerance_cms']==.5
    assert r['onset_direction']=='rise'
    # A brief excursion does not supply the required sustained onset.
    q[5]=5
    assert m.parent_reference(t,q,0.,0.,t[69])['t0_seconds']==t[20]
    with pytest.raises(ValueError):m.parent_reference(t,q,0.,0.,t[10])


def test_experiments_remain_separate():
    specs=m.specifications();assert len(specs)==72
    for s in specs:
        if s['variant']=='normalized_timing':
            assert s['target']==m.PCT
            assert not any(c in s['columns'] for c in ['B_mean_length_fraction','B_exposed_length_change','B_exposed_sd_change'])
        else:assert s['target']==m.TARGET
        if s['variant']=='length_context':
            assert 'B_mean_length_fraction' in s['columns'] and 'B_exposed_length_change' in s['columns']
            assert not any(c.endswith('B_ratio_route_length_mean') for c in s['columns'])
        if s['variant']=='spread_context':assert 'B_exposed_sd_change' in s['columns']
        assert not any('peak' in c or 't0' in c for c in s['columns'])


def test_percentage_backconversion_resume_and_reports(tmp_path,monkeypatch):
    d=pd.DataFrame([dict(example=str(ex),state=str(i),eligible=True,comparison='partial collapse',
      peak_shift_hours=float(i),peak_shift_parent_rise_pct=100*i/(ex-8),parent_time_to_peak_hours=float(ex-8))
      for ex in range(10,15) for i in [0,1]])
    calls=[]
    def choose(tr,target,spec,jobs):
        calls.append(target);best=dict(params={},mse=.1);return best,[best]
    def fit(tr,te,target,spec,params,jobs):
        assert set(tr.example).isdisjoint(te.example)
        return te[target].to_numpy(),dict(kept=[],dropped={})
    monkeypatch.setattr(m.engine,'choose',choose);monkeypatch.setattr(m.engine,'fit_predict',fit)
    specs=[s for s in m.specifications() if s['name'] in ['baseline__sd__raw__ridge','normalized_timing__sd__raw__ridge']]
    design=dict(specifications=specs)
    args=SimpleNamespace(experiments=['normalization'],models=['ridge'],scenarios=['all'],jobs=1)
    m.run(d,design,tmp_path,args);m.run(d,design,tmp_path,args)
    assert len(calls)==10
    pred=pd.read_parquet(tmp_path/'predictions.parquet')
    z=pred[pred.variant!='reference']
    np.testing.assert_allclose(z.observed_hours,z.predicted_hours)
    np.testing.assert_allclose(z.observed_pct,z.predicted_pct)
    assert set(pd.read_csv(tmp_path/'scores.csv').unit)=={'hours','pct'}
