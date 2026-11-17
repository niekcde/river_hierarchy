import numpy as np
import pandas as pd
import pytest
from prototype import engine,parameters,route,r

@pytest.fixture(scope='module')
def fun():return engine()

def frame():
 return pd.DataFrame([dict(u=0,v=1,id_link=1,count=3,link_length_m=3000.,rapid_effective_length_m=1000.,width=100.,slope=.0001,
    reference_discharge_cms=100.,reference_flow_fraction=1.,split_fraction=1.)])

@pytest.mark.parametrize('beta',[.5,1.,2.])
def test_constant_coefficients_match_production(fun,beta):
 f=parameters(frame(),.035,beta);q=100+50*np.sin(np.linspace(0,np.pi,180))**2
 got,_,_=route(fun,f,q,15,100,beta,False);expected,_=r.solve_wave(f,q,15,100)
 np.testing.assert_allclose(got,expected,rtol=1e-11,atol=1e-9)

def test_steady_and_invalid_flow(fun):
 f=parameters(frame(),.035,1.)
 q,a,_=route(fun,f,np.full(60,100.),15,100)
 np.testing.assert_allclose(q,100,atol=1e-9,rtol=0)
 assert abs(a['storage_balance_residual_m3'])<1e-6
 with pytest.raises(ValueError,match='positive flow'):route(fun,f,np.array([100.,0.]),15,100)

def test_branch_merge_conserves_steady_allocation(fun):
 f=frame().iloc[[0,0,0,0]].copy().reset_index(drop=True)
 f['u']=[0,0,1,2];f['v']=[1,2,3,3];f['id_link']=[1,2,3,4]
 f['split_fraction']=[.4,.6,1.,1.];f['reference_flow_fraction']=[.4,.6,.4,.6]
 f['reference_discharge_cms']=100*f.reference_flow_fraction
 q,a,_=route(fun,parameters(f,.035,1.),np.full(60,100.),15,100)
 np.testing.assert_allclose(q,100,atol=1e-9,rtol=0)
 assert abs(a['storage_balance_residual_m3'])<1e-6
