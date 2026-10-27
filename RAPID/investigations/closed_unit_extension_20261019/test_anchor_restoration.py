import numpy as np
from shapely.geometry import LineString,Point
from prepare_all import restore_endpoints,reparameterize,pilot
import pandas as pd

def test_restores_known_anchors_preserving_interior():
    line=LineString([(1,0),(40,10),(99,0)])
    new,offsets=restore_endpoints(line,Point(0,0),Point(100,0))
    assert tuple(new.coords[0])==(0,0) and tuple(new.coords[-1])==(100,0)
    assert tuple(new.coords[1])==tuple(line.coords[1]);np.testing.assert_allclose(offsets,[1,1])

def test_reparameterization_preserves_head_drop_and_calibration():
    f=pd.DataFrame([dict(id_link=0,u=0,v=1,id_us_node=0,id_ds_node=1,link_length_m=1000.,width=100.,depth_m=2.,head_us=11.,head_ds=10.,slope=.001)])
    old=pilot.cunge.parameters(pilot.cunge.freeze_grid(f,100),.04,beta=.5)
    new=reparameterize(old,1100.)
    np.testing.assert_allclose(new.slope*new.link_length_m,1.)
    for c in ['n_manning','diffusion_multiplier','width','depth_m','head_us','head_ds','reference_discharge_cms']:np.testing.assert_allclose(old[c],new[c])
    assert new.rapid_celerity_mps.iloc[0]<old.rapid_celerity_mps.iloc[0]
