import numpy as np
import pandas as pd
import pytest
from rapid_tools.celerity_regularization import regularize_routing_celerity


def sample():
    return pd.DataFrame({'rapid_effective_length_m':[200.,200.,220.],
                         'rapid_celerity_mps':[.1,5.,1.],
                         'rapid_celerity_mps_raw':[.1,6.,1.],
                         'slope_used':[1e-5,1e-3,1e-4]})


def test_preserves_hydraulic_inputs_and_opens_interior_interval():
    source=sample(); before=source.copy(deep=True)
    out=regularize_routing_celerity(source,dt_seconds=100)
    pd.testing.assert_frame_equal(source,before)
    for col in ['rapid_effective_length_m','rapid_celerity_mps_raw','slope_used']:
        pd.testing.assert_series_equal(source[col],out[col])
    assert out.rapid_celerity_regularization_reason.tolist()==[
        'slow_link_routing_floor','fast_link_routing_ceiling','not_required']
    assert np.allclose(out.rapid_celerity_mps,[.404,3.564,1.])
    assert (1.01*.2*out.rapid_k<=100+1e-10).all()
    assert (.99*1.8*out.rapid_k>=100-1e-10).all()


def test_same_link_same_clock_same_correction_in_parent_and_child():
    parent=regularize_routing_celerity(sample(),dt_seconds=100)
    child=regularize_routing_celerity(sample().iloc[[0,2]],dt_seconds=100)
    pd.testing.assert_series_equal(parent.loc[[0,2],'rapid_celerity_mps'],child.rapid_celerity_mps)


@pytest.mark.parametrize('kwargs',[{'dt_seconds':0},{'dt_seconds':1.5},
                                  {'dt_seconds':1},{'dt_seconds':100,'margin':.9}])
def test_rejects_invalid_or_cap_incompatible_clock(kwargs):
    with pytest.raises(ValueError): regularize_routing_celerity(sample(),**kwargs)


def test_repeated_regularization_cannot_hide_original_correction():
    out=regularize_routing_celerity(sample(),dt_seconds=100)
    with pytest.raises(ValueError,match='Already regularized'):
        regularize_routing_celerity(out,dt_seconds=110)
