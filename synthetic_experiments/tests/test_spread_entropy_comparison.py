import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'investigations'))
from spread_entropy_comparison import differences

def row(sd=5000.,mean=15000.,entropy=np.log(2)):
    return dict(B_parent_route_length_sd=sd,B_child_route_length_sd=0.,B_parent_route_length_mean=mean,B_parent_route_entropy=entropy,B_child_route_entropy=0.)

def test_full_collapse_differences_retain_magnitude():
    a=differences(row());assert a['route_spread_change_m']==-5000.
    assert np.isclose(a['route_spread_change_over_parent_mean'],-1/3)
    assert np.isclose(a['route_entropy_change'],-np.log(2))

def test_equal_routes_and_geometric_scale():
    a=differences(row(sd=0));assert a['route_spread_change_m']==0
    a=differences(row());b=differences(row(),2)
    assert b['route_spread_change_m']==2*a['route_spread_change_m']
    assert b['route_spread_change_over_parent_mean']==a['route_spread_change_over_parent_mean']
    assert b['route_entropy_change']==a['route_entropy_change']
