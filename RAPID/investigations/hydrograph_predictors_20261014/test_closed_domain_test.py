import numpy as np
import pandas as pd
import pytest
import closed_domain_test as m


def test_all_local_features_move_together_only_for_closed_units():
    d=pd.DataFrame({c:[9.,9.,9.] for c in m.forest.sweep.COLUMNS})
    d['U_closed']=[True,False,None]
    for metric in m.METRICS:
        d['U_parent_'+metric]=[2.,2.,2.]
        d['U_child_'+metric]=[1.,1.,1.]
    d['U_parent_route_length_mean']=4000.;d['U_child_route_length_mean']=2000.
    d['U_parent_dominance_mean']=.7;d['U_exposure']=.25
    a,use=m.local_inputs(d)
    assert use.tolist()==[True,False,False]
    for metric in m.METRICS:assert a.loc[0,'log1p__B_ratio_'+metric]==pytest.approx(np.log1p(.5))
    assert a.loc[0,'domain_size']==4.
    assert a.loc[0,'B_parent_dominance_mean']==.7
    assert a.loc[0,'B_exposure']==.25
    pd.testing.assert_frame_equal(a.iloc[1:],d.iloc[1:])
    d.loc[0,'U_parent_route_entropy']=0
    with pytest.raises(ValueError,match='Undefined'):m.local_inputs(d)
