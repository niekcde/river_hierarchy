import numpy as np
import pandas as pd
import pytest
from edited_features import describe_edited,pilot


def example(length=200,width=20):
    f=pd.DataFrame([(0,0,1,100,2),(1,0,1,200,1),(2,1,2,length,width),(3,2,3,150,1),(4,2,3,300,3)],columns=['id_link','id_us_node','id_ds_node','link_length_m','width'])
    return f


def test_full_mask_matches_established_definitions():
    f=example();new=describe_edited(f,f.id_link);old=pilot.descriptors(f)
    for k in new:np.testing.assert_allclose(new[k],old[k],rtol=1e-10,atol=1e-8)


def test_connector_length_and_width_do_not_enter_masked_predictors():
    a=describe_edited(example(),[0,1,3,4]);b=describe_edited(example(90000,5000),[0,1,3,4])
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=1e-12)
    np.testing.assert_allclose(a['route_length_mean'],(100*2+200)/3+(150+900)/4)
    np.testing.assert_allclose(a['route_length_sd']**2,100**2*2/9+150**2*3/16)


def test_partial_split_mask_rejected():
    with pytest.raises(ValueError,match='split'):describe_edited(example(),[0,3,4])
