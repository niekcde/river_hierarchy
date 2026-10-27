import json
import numpy as np
import pandas as pd
import pytest
import buffered_two_stage as m


def sample():
    rng=np.random.default_rng(42)
    d=pd.DataFrame(rng.uniform(.1,2,(100,len(m.sweep.COLUMNS))),columns=m.sweep.COLUMNS)
    d['example']=np.repeat(['01','02','03','04','05'],20)
    d['state']=[f'S{i}' for i in range(len(d))]
    d[m.TARGET]=np.where(d.domain_size>1.1,(d.domain_size-1.1)/4,0.)
    d['peak_shift_hours']=d[m.TARGET]*1.1
    d['edit_type']='Simple'
    return d


def test_buffer_boundary_and_gate():
    np.testing.assert_array_equal(m.changed(np.array([-2,-1,0,1,2])/60,1),[True,False,False,False,True])
    np.testing.assert_allclose(m.gate([.2,.5,.8],[-2,3,-4],.5),[0,3,-4])


def test_constant_classifier_and_empty_changed_subset():
    d=sample();d[m.TARGET]=0
    p,_=m.classifier_predict(d,d,1,m.CLASSIFIERS[0],1)
    q,a=m.regressor_predict(d,d,1,m.REGRESSORS[0],1)
    assert (p==0).all() and (q==0).all() and a['no_changed_training_rows']
    d[m.TARGET]=1
    p,_=m.classifier_predict(d,d,1,m.CLASSIFIERS[0],1)
    assert (p==1).all()


def test_outer_targets_do_not_affect_predictions(monkeypatch):
    monkeypatch.setattr(m,'CLASSIFIERS',m.CLASSIFIERS[:1])
    monkeypatch.setattr(m,'REGRESSORS',m.REGRESSORS[:1])
    d=sample();tr=d[d.example!='05'];te=d[d.example=='05'].copy()
    a=m.fit_fold(tr,te,1,1)
    te[m.TARGET]=999
    b=m.fit_fold(tr,te,1,1)
    assert a['best']==b['best']
    for column in ['predicted_hours','changed_probability','conditional_prediction_hours']:
        np.testing.assert_allclose([r[column] for r in a['predictions']],[r[column] for r in b['predictions']])
    assert a['regressor_audit']['changed_training_rows']==m.changed(tr[m.TARGET],1).sum()


def test_selection_uses_fixed_floor_and_network_weights():
    d=pd.DataFrame({'example':['01','02','02'],m.TARGET:np.array([0,2,2])/60})
    probs=np.array([[.3,.8,.8]])
    reg=np.array([[3,2,2]])/60
    best,trials=m.select(d,5,probs,reg)
    assert best['score']==pytest.approx(0)
    assert best['threshold']>.3
    assert next(t for t in trials if t['threshold']==.2)['score']==pytest.approx(150.)


def test_summary_and_checkpoint_guard(tmp_path):
    d=sample();te=d[d.example=='05']
    base=te[['example','state']].copy();base['scenario']='all';base['predicted_hours']=te[m.TARGET].to_numpy()
    rows=[dict(example=r.example,state=r.state,observed_hours=getattr(r,m.TARGET),predicted_hours=getattr(r,m.TARGET),changed_probability=1.,conditional_prediction_hours=getattr(r,m.TARGET),gate_changed=True) for r in te.itertuples()]
    m.engine.dump(tmp_path/'checkpoints/all/05__buffer1.json',dict(fingerprint='test',scenario='all',held='05',buffer_minutes=1,predictions=rows))
    m.summarize(tmp_path,d,base,'test')
    scores=pd.read_csv(tmp_path/'scores.csv')
    assert not scores.complete.any()
    assert (scores[(scores.response==m.TARGET)&(scores.subset=='all')].mae==0).all()
    with pytest.raises(AssertionError):m.summarize(tmp_path,d,base,'wrong')
