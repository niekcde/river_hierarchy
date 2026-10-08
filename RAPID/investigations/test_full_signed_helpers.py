"""Small analytical checks for new-node allocation and frequency transfer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_signed_routing_validation import *
from route_full_signed_validation import response
from check_explicit_celerity_regularization import transfer

def test_chain_heads():
 f=pd.DataFrame(dict(u=[0,1],v=[1,2],link_length_m=[100.,200.]))
 h=allocate_new_heads(f,pd.Series({0:3.,2:0.}),{1})
 assert np.isclose(h.loc[1],2.) and h.loc[0]==3 and h.loc[2]==0

def test_infeasible_heads():
 f=pd.DataFrame(dict(u=[0,1],v=[1,2],link_length_m=[100.,200.]))
 try:allocate_new_heads(f,pd.Series({0:0.,2:3.}),{1})
 except ValueError:pass
 else:raise AssertionError('Infeasible boundary heads accepted')

def test_frequency_transfer():
 f=pd.DataFrame(dict(u=[0,1,1,2,3],v=[1,2,3,4,4],width=[10.,3.,7.,3.,7.],count=[1,2,1,1,1],rapid_k=[100.,120.,900.,130.,800.]))
 dt=10.;n=4096;w=2*np.pi*np.arange(n//2+1)/(n*dt)
 assert np.allclose(response(f,dt,n,dt),transfer(f,dt,n)*np.sinc(w*dt/(2*np.pi)),rtol=1e-10,atol=1e-12)

if __name__=='__main__':
 test_chain_heads();test_infeasible_heads();test_frequency_transfer();print('3 analytical checks passed')
