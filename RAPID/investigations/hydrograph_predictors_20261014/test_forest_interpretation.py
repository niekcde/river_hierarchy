import numpy as np
import pandas as pd
import pytest
import forest_interpretation as m


def sample():
    rng=np.random.default_rng(31)
    d=pd.DataFrame(rng.uniform(.1,2,(60,len(m.PANELS['plus_resistance']))),columns=m.PANELS['plus_resistance'])
    d['example']=np.repeat(['01','02','03'],20)
    d['state']=[str(i) for i in range(len(d))]
    d[m.TARGET]=d.domain_size/10-d[m.WIDTH]/8
    d['peak_shift_hours']=d[m.TARGET]*1.1
    return d


def test_forest_reconstruction_matches_sweep():
    d=sample();tr,te=d.iloc[:40],d.iloc[40:].copy();params=m.sweep.GRIDS['forest'][0]
    p=m.predict(m.fit(tr,m.PANELS['current'],params,1),te)
    q,_=m.sweep.fit_predict(tr,te,m.TARGET,'forest',params,1)
    np.testing.assert_allclose(p,q)
    te[m.TARGET]=1e6
    np.testing.assert_allclose(p,m.predict(m.fit(tr,m.PANELS['current'],params,1),te))


def test_panels_change_only_specified_columns():
    assert m.PANELS['plus_resistance']==m.PANELS['current']+[m.RESISTANCE]
    assert set(m.PANELS['resistance_replaces_width'])==(set(m.PANELS['current'])-{m.WIDTH})|{m.RESISTANCE}
    assert all(set(cols)<=set(m.sweep.COLUMNS) for cols in m.GROUPS.values())


def test_tree_shap_additivity_if_available():
    shap=pytest.importorskip('shap')
    d=sample();pre,forest,mean,sd=m.fit(d.iloc[:40],m.PANELS['current'],m.sweep.GRIDS['forest'][0],1)
    x=pre.transform(d.iloc[40:]);exp=shap.TreeExplainer(forest,feature_perturbation='tree_path_dependent')
    sv=exp.shap_values(x);iv=exp.shap_interaction_values(x)
    np.testing.assert_allclose(np.asarray(sv).sum(axis=1)+float(exp.expected_value[0]),forest.predict(x),atol=1e-8)
    np.testing.assert_allclose(np.asarray(iv).sum(axis=2),sv,atol=1e-8)
