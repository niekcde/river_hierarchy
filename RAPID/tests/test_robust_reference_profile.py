import numpy as np
import pandas as pd
import pytest
from rapid_tools.robust_reference_profile import (
    disconnected_gap_anchors, prepare_problem, fit_robust_profile,
    score_held_cycles,
)


def test_gap_anchors_fix_unobserved_head_difference():
    x=np.arange(6)*1000.
    profile=pd.DataFrame({'reference_section_dist_out_m':x,'static_wse_m':x*.001})
    obs=pd.DataFrame({'reference_section_dist_out_m':[0.,1000.,4000.,5000.],
                      'wse_m':[20.,21.,80.,81.], 'acquisition_id':['a','a','b','b'],
                      'cycle':[1,1,2,2]})
    gaps=disconnected_gap_anchors(profile,obs,.0001)
    assert len(gaps)==1 and gaps[0]['drop_m']==3
    p=prepare_problem(profile,obs,.002,gaps=gaps)
    h,v,a=fit_robust_profile(p,1.,.1)
    np.testing.assert_allclose(h[4]-h[1],3,atol=1e-5)
    np.testing.assert_allclose(np.diff(h),1,atol=1e-4)
    assert a['constraint_violation_m']<1e-5
    shifted=obs.copy();shifted.loc[shifted.acquisition_id=='b','wse_m']+=200
    pp=prepare_problem(profile,shifted,.002,gaps=gaps)
    hh,_,_=fit_robust_profile(pp,1.,.1)
    np.testing.assert_allclose(hh-hh[0],h-h[0],atol=1e-4)


def test_pass_spanning_gap_does_not_get_static_anchor():
    p=pd.DataFrame({'reference_section_dist_out_m':[0.,1.,2.,3.], 'static_wse_m':[0.,1.,2.,3.]})
    o=pd.DataFrame({'reference_section_dist_out_m':[0.,3.], 'acquisition_id':['a','a']})
    assert disconnected_gap_anchors(p,o,.1)==[]


def test_incompatible_static_gap_raises():
    p=pd.DataFrame({'reference_section_dist_out_m':[0.,1.,2.,3.], 'static_wse_m':[0.,1.,.5,2.]})
    o=pd.DataFrame({'reference_section_dist_out_m':[0.,1.,2.,3.], 'acquisition_id':['a','a','b','b']})
    with pytest.raises(ValueError,match='incompatible'):
        disconnected_gap_anchors(p,o,.1)


def test_huber_reduces_single_bad_cycle_influence_and_scores_remove_stage():
    x=np.arange(5)*1000.;p=pd.DataFrame({'reference_section_dist_out_m':x,'static_wse_m':x*.001})
    o=pd.DataFrame([{'reference_section_dist_out_m':xx,'wse_m':xx*.001+(20 if cycle==4 and xx==2000 else 0), 'cycle':cycle,'acquisition_id':str(cycle)} for cycle in range(5) for xx in x])
    problem=prepare_problem(p,o,.001)
    hs,_,_=fit_robust_profile(problem,1.,np.inf)
    hr,_,_=fit_robust_profile(problem,1.,.1)
    truth=x*.001
    assert np.mean((hr-hr[0]-truth)**2)<np.mean((hs-hs[0]-truth)**2)
    good=o[o.cycle==0].copy();good.wse_m+=123
    score=score_held_cycles(good,x,truth,.1)
    assert score.to_numpy().max()<1e-20
