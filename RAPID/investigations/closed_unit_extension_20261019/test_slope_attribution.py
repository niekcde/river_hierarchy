import numpy as np
import pandas as pd
from slope_attribution import attribute,descriptor

def test_uniform_uncapped_log_effects_are_exact():
 s=4e-4;h=10.;n=.06;c=(5/3)*h**(2/3)*np.sqrt(s)/n
 f=pd.DataFrame(dict(reference_flow_fraction=[1.],link_length_m=[100.],slope=[s],depth_m=[h],n_manning=[n],celerity_min_mps=[1e-8],celerity_max_mps=[100.],rapid_celerity_mps=[c]))
 phi,speed=attribute(f,dict(slope=1e-4,depth=5.,roughness=.03))
 assert np.isclose(speed,c)
 assert np.isclose(phi['slope'],.5*np.log(4))
 assert np.isclose(phi['depth'],(2/3)*np.log(2))
 assert np.isclose(phi['roughness'],-np.log(2))
 assert abs(phi['cap'])<1e-12

def test_fixed_head_slope_ratio_redundancy_and_uniform_proxy():
 def f(L,p):return pd.DataFrame(dict(reference_flow_fraction=p,link_length_m=L,slope=10/np.asarray(L),depth_m=5.,n_manning=.03))
 parent=descriptor(f([100.,200.],[.25,.75]));child=descriptor(f([80.],[1.]))
 assert np.isclose(child['slope_mean']/parent['slope_mean'],parent['length_km']/child['length_km'])
 assert np.isclose(child['slope_proxy_km'],child['length_km']/np.sqrt(child['slope_mean']))
