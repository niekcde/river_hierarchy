import numpy as np
from build import mean_valid,component_masks,NODATA

def test_missing_observations_not_land():
    total=np.array([3,3,0],float);count=np.array([5,4,0])
    a=mean_valid(total,count)
    assert np.isclose(a[0],.6) and a[1]==NODATA and a[2]==NODATA

def test_valid_dry_pixels_are_zero():
    assert mean_valid(np.array([0.]),np.array([20]))[0]==0

def test_eight_connected_and_preserve_exclusions():
    a=np.zeros((20,20),bool);a[2:6,2:6]=1;a[6,6]=1;a[15:17,15:17]=1
    main,excluded,n=component_masks(a,np.ones_like(a))
    assert n==2 and main[6,6] and excluded.sum()==4
    assert np.array_equal(main|excluded,a) and not(main&excluded).any()

def test_invalid_cannot_bridge_components():
    a=np.zeros((10,10),bool);a[5,1:9]=1;valid=np.ones_like(a);valid[:,4:6]=0
    main,excluded,n=component_masks(a,valid)
    assert n==2 and not(main&~valid).any() and np.array_equal(main|excluded,a&valid)

def test_empty_water_is_not_all_water():
    a=np.zeros((10,10),bool);main,excluded,n=component_masks(a,np.ones_like(a))
    assert n==0 and not main.any() and not excluded.any()
