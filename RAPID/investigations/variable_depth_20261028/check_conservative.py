"""Storage-consistent diagnostic tests; no production calibration."""
import json
import numpy as np
import pandas as pd
from prototype import engine,route,parameters,DEST,r,HERE
from test_prototype import frame

def main():
 fun=engine('conservative.cpp');rows=[]
 for beta in [.5,1.,2.]:
  f=parameters(frame(),.035,beta)
  pulse=np.r_[100+50*np.sin(np.linspace(0,np.pi,720))**2,np.full(14400,100.)]
  steady,a,_=route(fun,f,np.full(60,100.),15,100,beta)
  np.testing.assert_allclose(steady,100,rtol=0,atol=1e-9)
  for dt in [15.,7.5]:
   q,a,_=route(fun,f,pulse,dt,100,beta)
   err=100*((q-100).sum()/(pulse-100).sum()-1)
   assert abs(q[-1]-100)<1e-6 and abs(err)<1e-6
   assert abs(a['residual_pct_excess_volume'])<1e-6
   rows.append(dict(beta=beta,dt=dt,closed_pulse_volume_error_pct=err,**a))
 # Different final equilibrium verifies stored water, not only closed-pulse totals.
 f=parameters(frame(),.035,1.);step=np.r_[np.linspace(100,150,720),np.full(14400,150.)]
 q,a,_=route(fun,f,step,15,100)
 expected_storage=3000*100*((.035*150/(100*.0001**.5))**.6-(.035*100/(100*.0001**.5))**.6)
 assert abs(a['physical_storage_change_m3']/expected_storage-1)<1e-8
 assert abs(a['residual_pct_excess_volume'])<1e-6
 rows.append(dict(test='changed_final_equilibrium',beta=1.,dt=15,**a))
 pd.DataFrame(rows).to_csv(DEST/'conservative_implementation_checks.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False),flush=True)
 r.write_json(DEST/'conservative_method.json',dict(status='synthetic_checks_passed_real_network_validation_pending',
 method='Implicit conservative trapezoidal continuity with Todini corrected storage; positive-root bracketed Newton correction.',
 storage='S(I,O)=dx*B*h((I+O)/2) + beta*Q/(2*B*slope*c(Q)^2)*(O-I), Q=(I+O)/2',
 continuity='S_new-S_old = dt/2*(I_new+I_old-O_new-O_old)',
 qualification='beta is our diffusion multiplier, NOT Todini c/v correction. Storage is the model prism-plus-wedge storage, not trapezoidal endpoint-area storage during transients. Existing timestep screen is preliminary and not a certification for this scheme.',
 reference='https://hess.copernicus.org/articles/11/1645/2007/',source_hashes={str(p):r.digest(p) for p in [HERE/'conservative.cpp',HERE/'check_conservative.py',HERE/'prototype.py']}))
if __name__=='__main__':main()
