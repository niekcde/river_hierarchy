"""Analytic validation of the new hydraulic and shared-route summaries."""
import numpy as np
import pandas as pd
from features import describe, checks


def hydraulic_frame():
    return pd.DataFrame(dict(id_link=[0,1,2],id_us_node=[0,0,1],id_ds_node=[1,1,2],link_length_m=[100.,200.,50.],width=[3.,1.,4.],slope=[.001]*3,depth_m=[2.]*3,rapid_celerity_mps=[2.]*3,hydraulic_diffusivity_m2_s=[3.]*3,rapid_celerity_capped=[False]*3))


def run():
    checks();f=hydraulic_frame();a,_=describe(f)
    np.testing.assert_allclose([a['route_time_mean'],a['route_time_sd']**2,a['within_route_dispersion'],a['total_time_spread']**2],[87.5,468.75,131.25,600.],rtol=1e-12)
    # Convert metres to kilometres, retaining seconds: all time moments and
    # dimensionless graph statistics must remain unchanged.
    scaled=f.copy()
    for c in ['link_length_m','width','depth_m','rapid_celerity_mps']:scaled[c]*=.001
    scaled.hydraulic_diffusivity_m2_s*=1e-6
    b,_=describe(scaled)
    for c in ['route_time_mean','route_time_sd','within_route_dispersion','total_time_spread','dominance_mean','expected_splits','route_length_cv']:np.testing.assert_allclose(a[c],b[c],rtol=1e-12)
    # Equal-length branches can still have unequal arrival times.
    uneven=f.copy();uneven.loc[1,'link_length_m']=100.;uneven.loc[1,'rapid_celerity_mps']=1.
    c,_=describe(uneven);assert c['route_length_sd']==0 and c['route_time_sd']>0
    # Two serial diamonds: four complete routes but only two branching encounters.
    serial=pd.DataFrame(dict(id_link=[0,1,2,3],id_us_node=[0,0,1,1],id_ds_node=[1,1,2,2],link_length_m=[100.]*4,width=[10.]*4))
    s,_=describe(serial,False);np.testing.assert_allclose([s['expected_splits'],s['dominance_mean'],s['split_impact_mean'],s['split_impact_sum'],s['route_length_cv']],[2.,.5,.5,1.,0.])
    print('PASS: branch moments, hydraulic dispersion, shared downstream geometry, SI unit conversion, unequal celerity, serial splitting, geometric subdivision and Π basis.')

if __name__=='__main__':run()
