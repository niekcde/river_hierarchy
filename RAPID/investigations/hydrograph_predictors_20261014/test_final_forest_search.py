import json
import joblib
import numpy as np
import pandas as pd
import pytest
import final_forest_search as m


def sample():
    rng=np.random.default_rng(12)
    d=pd.DataFrame(rng.uniform(.1,2,(72,len(m.sweep.COLUMNS))),columns=m.sweep.COLUMNS)
    d['example']=np.repeat(['01','02','03','04'],18)
    d['state']=[f'S{i}' for i in range(len(d))]
    d[m.TARGET]=d.domain_size/8-d.log1p__B_ratio_width_mean/6
    d['peak_shift_hours']=d[m.TARGET]
    return d


def small_grids():
    return [dict(p,n_estimators=4,min_samples_leaf=2) for p in [m.candidates()[0],m.candidates()[-1]]]


def test_grid_unique_valid_and_preserves_prior_settings():
    grid=m.candidates()
    assert grid==m.candidates() and len(grid)==96
    assert len({json.dumps(p,sort_keys=True) for p in grid})==96
    assert sum(p['criterion']=='absolute_error' for p in grid)==16
    for p in grid:
        assert p['bootstrap'] or p['max_samples'] is None
        assert p['min_samples_split']==2 or p['min_samples_split']>2*p['min_samples_leaf']
    for p,q in zip(grid,m.sweep.GRIDS['forest']):
        assert all(p[k]==v for k,v in q.items())
        assert p['n_estimators']==200 and p['criterion']=='squared_error'


def test_old_setting_reproduced_and_test_target_ignored():
    d=sample();tr,te=d.iloc[:54],d.iloc[54:].copy()
    artifact=m.fit(tr,m.candidates()[0],1);p=m.predict(artifact,te)
    ref,_=m.sweep.fit_predict(tr,te,m.TARGET,'forest',m.sweep.GRIDS['forest'][0],1)
    np.testing.assert_allclose(p,ref)
    te[m.TARGET]=999
    np.testing.assert_allclose(m.predict(artifact,te),p)


def test_group_tuning_resumes_and_guards_fingerprint(tmp_path,monkeypatch):
    d=sample();grids=small_grids();seen=[];original=m.fit
    def fitting(tr,params,jobs):
        seen.append(set(tr.example))
        return original(tr,params,jobs)
    monkeypatch.setattr(m,'fit',fitting)
    best,trials=m.tune(d,grids,tmp_path/'tuning/all/01','abc',1,keep_predictions=True)
    assert len(seen)==6
    assert all(len(ex)<d.example.nunique() for ex in seen)
    for t in trials:
        a=pd.DataFrame(t['predictions']);counts=a.example.value_counts()
        assert counts.to_dict()==d.example.value_counts().to_dict()
        expected=m.sweep.relative_score(a.observed_hours,a.predicted_hours,m.engine.weights(a),1)
        assert t['selection_score']==pytest.approx(expected)
    again,_=m.tune(d,grids,tmp_path/'tuning/all/01','abc',1,keep_predictions=True)
    assert len(seen)==6 and again==best
    with pytest.raises(ValueError,match='Incompatible'):
        m.tune(d,grids,tmp_path/'tuning/all/01','different',1)


def test_final_selection_and_saved_model(tmp_path):
    d=sample();grids=small_grids()
    m.finalize(d,grids,tmp_path,'abc',1,dict(versions={'test':'fixture'}))
    info=json.loads((tmp_path/'final_model.json').read_text())
    artifact=joblib.load(tmp_path/'final_model.joblib')
    assert info['params']==artifact['params'] and info['training_rows']==len(d)
    assert artifact['training_examples']==sorted(d.example.unique())
    assert np.isfinite(m.predict(artifact,d)).all()
    before=(tmp_path/'final_model.joblib').read_bytes()
    m.finalize(d,grids,tmp_path,'abc',1,dict(versions={'test':'fixture'}))
    assert before==(tmp_path/'final_model.joblib').read_bytes()
    frame=d.copy();frame.loc[0,m.sweep.COLUMNS[0]]=np.nan
    with pytest.raises(ValueError,match='Finite'):m.predict(artifact,frame)


def test_serialization_tolerance_accepts_roundoff_rejects_changes(monkeypatch):
    monkeypatch.setattr(m,'predict',lambda a,frame:np.asarray(a))
    m.serialization_check([1.,0.],[1.+2e-16,2e-16],None)
    with pytest.raises(AssertionError):m.serialization_check([1.,0.],[1.,1e-6],None)
    with pytest.raises(AssertionError):m.serialization_check([np.nan],[np.nan],None)


def test_legacy_resume_preserves_fingerprint_rejects_input_changes(tmp_path):
    key=str(m.Path(m.__file__))
    old={'sources':{key:m.LEGACY_SEARCH_SOURCE,'input':'unchanged'},'settings':[1,2]}
    m.engine.dump(tmp_path/'design.json',old)
    m.engine.dump(tmp_path/'tuning/all/01/candidate_000.json',{'fingerprint':'old'})
    new={'sources':{key:'new_source','input':'unchanged'},'settings':[1,2]}
    assert m.resume_design(tmp_path,new)==old
    assert json.loads((tmp_path/'design.json').read_text())==old
    new['sources']['input']='changed'
    with pytest.raises(ValueError,match='differs'):m.resume_design(tmp_path,new)
