import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import focused_models as f


def test_fixed_inputs_and_logs():
    specs=f.specifications()
    assert len(specs)==16 and sum(len(s['grid']) for s in specs)==96
    for s in specs:
        assert len(s['columns'])==8 and len(set(s['columns']))==8
        assert s['corr'] is None and s['top_k'] is None
        assert set(f.CONTEXT)<=set(s['columns'])
        assert sum('route_length_sd' in c or 'route_length_cv' in c for c in s['columns'])==1
    d=pd.DataFrame({c:[0.,1.,3.] for c in f.RATIOS})
    for c in f.CONTEXT:d[c]=[-1.,0.,1.]
    t=f.transform(d)
    for c in f.RATIOS:np.testing.assert_allclose(t['log1p__'+c],np.log([1.,2.,4.]))
    pd.testing.assert_frame_equal(t[f.CONTEXT],d[f.CONTEXT])


def test_resuming_and_complete_reports(tmp_path,monkeypatch):
    d=pd.DataFrame([dict(example=str(ex),state=str(i),peak_shift_hours=float(i),eligible=True,
                        comparison='partial collapse') for ex in range(10,15) for i in range(3)])
    calls=[]
    def choose(tr,target,spec,jobs):
        calls.append(spec['name']);v=dict(params={},mse=.1);return v,[v]
    def fit(tr,te,target,spec,params,jobs):
        assert set(tr.example).isdisjoint(te.example)
        return np.full(len(te),tr[target].mean()),dict(kept=['x'],dropped={},warnings=[])
    monkeypatch.setattr(f.engine,'choose',choose)
    monkeypatch.setattr(f.engine,'fit_predict',fit)
    alt=pd.concat([d[['example','state','peak_shift_hours']].assign(window=w) for w in ['1%','2%']])
    monkeypatch.setattr(f.pd,'read_parquet',lambda p:alt)
    design={'specifications':[dict(name='sd__raw__ridge'),dict(name='cv__log1p__ridge')]}
    args=SimpleNamespace(cohorts=['partial'],scenarios=['all'],targets=['peak_shift_hours'],jobs=1)
    f.run(d,design,tmp_path,args)
    assert len(calls)==10
    f.run(d,design,tmp_path,args)
    assert len(calls)==10
    scores=pd.read_csv(tmp_path/'scores.csv')
    assert set(scores.pipeline)=={'sd__raw__ridge','cv__log1p__ridge','zero','training_mean'}
    assert scores[scores.weighting.eq('equal_example')].n.eq(15).all()
    assert len(pd.read_csv(tmp_path/'sensitivity_scores.csv'))==8


def test_constant_exposure_is_removed_but_not_replaced():
    rng=np.random.default_rng(1)
    d=pd.DataFrame({'example':np.repeat(['01','02','03'],10),'ratio':rng.uniform(0,2,30),'exposure':1.})
    pre=f.engine.Preprocess(['ratio','exposure']).fit(d,np.arange(30))
    assert pre.kept==['ratio']
    assert pre.dropped['exposure']=='training constant or empty'
