import numpy as np
import pytest
from prototype import engine,route,parameters
from test_prototype import frame

@pytest.fixture(scope='module')
def fun():return engine('conservative_v2.cpp')

@pytest.mark.parametrize('beta',[.5,1.,2.])
def test_pulse_preserves_volume(fun,beta):
 f=parameters(frame(),.035,beta)
 flow=np.r_[100+50*np.sin(np.linspace(0,np.pi,720))**2,np.full(14400,100.)]
 q,a,_=route(fun,f,flow,15,100,beta)
 assert abs(q[-1]-100)<1e-6
 assert abs((q-100).sum()/(flow-100).sum()-1)<1e-8
 assert abs(a['residual_pct_excess_volume'])<1e-6

def test_prism_storage_at_changed_equilibrium(fun):
 f=parameters(frame(),.035,1.)
 flow=np.r_[np.linspace(100,150,720),np.full(14400,150.)]
 q,a,_=route(fun,f,flow,15,100)
 expected=3000*100*((.035*150/(100*.0001**.5))**.6-(.035*100/(100*.0001**.5))**.6)
 assert abs(a['physical_storage_change_m3']/expected-1)<1e-8
 assert abs(a['residual_pct_excess_volume'])<1e-6

def test_branch_merge_pulse_conserves(fun):
 f=frame().iloc[[0,0,0,0]].copy().reset_index(drop=True)
 f['u']=[0,0,1,2];f['v']=[1,2,3,3];f['id_link']=[1,2,3,4]
 f['split_fraction']=[.4,.6,1.,1.];f['reference_flow_fraction']=[.4,.6,.4,.6]
 f['reference_discharge_cms']=100*f.reference_flow_fraction
 flow=np.r_[100+50*np.sin(np.linspace(0,np.pi,720))**2,np.full(14400,100.)]
 q,a,_=route(fun,parameters(f,.035,1.),flow,15,100)
 assert abs(q[-1]-100)<1e-6
 assert abs(a['residual_pct_excess_volume'])<1e-6


def test_prior_valid_waveform_unchanged(fun):
 f=parameters(frame(),.035,1.)
 flow=np.r_[100+50*np.sin(np.linspace(0,np.pi,120))**2,np.full(3600,100.)]
 q,a,_=route(fun,f,flow,15,100.)
 old,_,_=route(engine('conservative.cpp'),f,flow,15,100.)
 np.testing.assert_allclose(q,old,rtol=1e-10,atol=1e-8)
 assert abs(a['residual_pct_excess_volume'])<1e-6
