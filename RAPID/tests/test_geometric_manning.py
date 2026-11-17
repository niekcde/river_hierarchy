import numpy as np
import pandas as pd
from rapid_tools.geometric_manning import manning_geometry_edges,manning_geometry_factor

def frame(edges):
    f=pd.DataFrame(edges,columns=['u','v','link_length_m','width','slope']);f['id_link']=np.arange(len(f));return f

def test_equal_length_width_split_and_subdivision():
    p=frame([(0,1,10000,100,.0001),(0,1,10000,400,.0001)])
    c=frame([(0,1,10000,500,.0001)])
    split=frame([(0,2,5000,100,.0001),(2,1,5000,100,.0001),(0,1,10000,400,.0001)])
    np.testing.assert_allclose(manning_geometry_factor(p),manning_geometry_factor(c))
    np.testing.assert_allclose(manning_geometry_factor(p),manning_geometry_factor(split))

def test_reference_travel_identity_and_hydraulic_exclusion():
    p=frame([(0,1,5000,500,.0001),(1,2,15000,100,.0001),(1,2,10000,400,.0002),(2,3,5000,500,.0001)])
    e=manning_geometry_edges(p);n=.035;Q=750
    h=(n*Q*e.geometry_flow_fraction/(e.width*np.sqrt(e.slope)))**.6
    c=5/(3*n)*h**(2/3)*np.sqrt(e.slope)
    expected=np.sum(e.geometry_flow_fraction*e.link_length_m/c)
    np.testing.assert_allclose(.6*n**.6*Q**(-.4)*manning_geometry_factor(p),expected)
    changed=p.assign(depth_m=999,n_manning=555,reference_flow_fraction=0,discharge=123)
    assert manning_geometry_factor(changed)==manning_geometry_factor(p)

def test_surviving_reach_allocation_is_recomputed():
    p=frame([(0,1,1000,100,.001),(0,2,1000,100,.001),(1,3,1000,100,.001),(2,3,1000,100,.001)])
    c=p.copy();c.loc[0,'width']=200
    a,b=manning_geometry_edges(p),manning_geometry_edges(c)
    assert a.loc[2,'geometry_flow_fraction']==.5
    np.testing.assert_allclose(b.loc[2,'geometry_flow_fraction'],2/3)
    assert b.loc[2,'G_manning_contribution']>a.loc[2,'G_manning_contribution']
