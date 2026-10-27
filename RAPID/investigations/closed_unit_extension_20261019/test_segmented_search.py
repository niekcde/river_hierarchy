import numpy as np
import pandas as pd
import segmented_search as s

def test_objective_balances_networks_and_cohorts():
    d=pd.DataFrame({'example':['A','A','B','A','B'],'cohort':['original','original','original','segmented','segmented'],'centroid_shift_hours':[1.]*5})
    # Original A=0%, B=100%; segmented A=B=200% => mean(50,200)=125%.
    score,by=s.objective(d,np.array([1.,1.,2.,3.,3.]))
    assert np.isclose(score,125.) and np.isclose(by['original'],50.)
    doubled=pd.concat([d,d.iloc[[0,1]]],ignore_index=True)
    assert np.isclose(s.objective(doubled,np.array([1.,1.,2.,3.,3.,1.,1.]))[0],125.)

def test_edited_transform_changes_only_size():
    d=pd.DataFrame({'domain_size':[4.,5.],'domain_size_edited':[1.,2.],'other':[8.,9.]})
    actual=s.transformed(d,'edited');assert actual.domain_size.tolist()==[1.,2.]
    pd.testing.assert_frame_equal(actual.drop(columns='domain_size'),d.drop(columns='domain_size'))
    assert d.domain_size.tolist()==[4.,5.]

def test_candidates_include_fixed_control_for_each_size():
    params={'n_estimators':500,'criterion':'absolute_error','min_samples_leaf':2,'min_samples_split':20,'max_depth':16,'max_features':1.,'ccp_alpha':.001,'bootstrap':True,'max_samples':None}
    grids=s.candidates(params);assert len(grids)==24
    for size in ['current','none','edited']:
        selected=[r for r in grids if r['size']==size];assert len(selected)==8
        assert selected[0]['params']==params
