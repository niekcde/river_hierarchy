import numpy as np
import pandas as pd
import pytest
import relative_model_sweep as m


def test_relative_score_uses_hours_and_floor():
    y=np.array([100.,5.,0.])/60
    pred=np.array([95.,0.,.2])/60
    assert m.relative_score(y,pred,np.ones(3),1)==pytest.approx((5+100+20)/3)
    assert m.relative_score(y,pred,np.ones(3),5)==pytest.approx((5+100+4)/3)


def test_network_weighting_equalizes_network_sizes():
    d=pd.DataFrame(dict(example=['a','b','b','b']))
    y=np.ones(4)/60;p=np.array([0,1,1,1])/60
    assert m.relative_score(y,p,m.engine.weights(d),1)==pytest.approx(50)


@pytest.mark.parametrize('family',list(m.GRIDS))
def test_all_families_predict_without_using_test_targets(family):
    rng=np.random.default_rng(13)
    d=pd.DataFrame(rng.uniform(.1,2,(45,len(m.COLUMNS))),columns=m.COLUMNS)
    d['B_exposure']=1.
    d['example']=np.repeat(['01','02','03'],15)
    d['centroid_shift_hours']=.1*d.domain_size-.2*d.log1p__B_ratio_width_mean
    train=d.iloc[:30].copy();test=d.iloc[30:].copy()
    p,a=m.fit_predict(train,test,'centroid_shift_hours',family,m.GRIDS[family][0])
    test['centroid_shift_hours']=1e9
    q,_=m.fit_predict(train,test,'centroid_shift_hours',family,m.GRIDS[family][0])
    assert np.isfinite(p).all() and len(p)==15
    np.testing.assert_allclose(p,q)
    assert 'B_exposure' in a['dropped']


def test_width_first_and_separate_targets():
    assert 'B_K_log_a1' not in m.COLUMNS
    assert 'log1p__B_ratio_width_mean' in m.COLUMNS
    assert len(m.GRIDS)==8
    assert sum(map(len,m.GRIDS.values()))==48
    assert m.TARGETS[0]=='centroid_shift_hours'


def test_svr_retries_then_accepts_only_converged_fit():
    import warnings
    class Solver:
        def set_params(self,**kwargs):self.limit=kwargs['max_iter']
        def fit(self,*args,**kwargs):
            self.fit_status_=int(self.limit==100000)
            if self.fit_status_:warnings.warn('iteration limit',m.ConvergenceWarning)
    attempts=m.fit_converged(Solver(),None,None,None,'svr')
    assert [a['max_iter'] for a in attempts]==[100000,1000000]
    assert not attempts[0]['converged'] and attempts[1]['converged']


def test_svr_failure_status_is_rejected_even_without_warning():
    class Solver:
        def set_params(self,**kwargs):pass
        def fit(self,*args,**kwargs):self.fit_status_=1
    with pytest.raises(m.FitConvergenceError):m.fit_converged(Solver(),None,None,None,'svr')


def test_legacy_resume_preserves_checkpoint_and_rejects_changed_inputs(tmp_path):
    import json,hashlib
    from pathlib import Path
    script=str(Path(m.__file__))
    old=dict(sources={script:m.LEGACY_CONVERGED_SOURCE,'data':'original'},targets=['centroid'])
    fp=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    m.engine.dump(tmp_path/'design.json',old)
    checkpoint=tmp_path/'checkpoints/all/network/02/model.json'
    m.engine.dump(checkpoint,dict(fingerprint=fp,predictions=[1,2]))
    before=checkpoint.read_bytes()
    new=dict(sources={script:'patched','data':'original'},targets=['centroid'])
    migrated=m.resume_design(tmp_path,new)
    assert fp in m.valid_fingerprints(migrated)
    assert checkpoint.read_bytes()==before
    m.engine.dump(tmp_path/'design.json',migrated)
    assert m.resume_design(tmp_path,new)==migrated
    changed=dict(sources={script:'patched','data':'changed'},targets=['centroid'])
    with pytest.raises(ValueError,match='incompatible'):m.resume_design(tmp_path,changed)
