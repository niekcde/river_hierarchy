"""Fast report-interval outlet response of the exact native linear recurrence.

DFT hold/average alias formula, with a magnitude bound on omitted aliases.
Used only for constant X=.1 controls; periodic-tail error checked separately.
"""
import numpy as np
from rapid_tools import cunge

def transfer(frame,dt,omega,envelope=False):
 k=frame.rapid_k.to_numpy();x=frame.rapid_x.to_numpy();den=dt+2*k*(1-x)
 a=(dt-2*k*x)/den;b=(dt+2*k*x)/den;d=(2*k*(1-x)-dt)/den
 assert np.all((x>=0)&(x<=.5)) and np.all((d>=-1e-12)&(d<1))
 # For x<=.5 the individual |H| decreases from DC to Nyquist.
 assert np.all(a*b*(1+d*d)+d*(a*a+b*b)>=-1e-12)
 order,edges,root,end=cunge.topology(frame);sig={root:np.ones(len(omega),dtype=float if envelope else complex)};z=np.exp(-1j*omega)
 for u in order:
  val=sig.pop(u)
  if u==end:return val
  for i in edges[u]:
   row=frame.iloc[i];h=(a[i]+b[i]*z)/(1-d[i]*z)
   if envelope:h=abs(h)
   v=val*row.split_fraction*h**int(row['count']);sig[row.v]=sig.get(row.v,0)+v
 raise ValueError('No outlet')

def route_frequency(frame,flow,nsub,dt,padding=4,tolerance_cms=1e-7):
 count=1<<int(np.ceil(np.log2(len(flow)*padding)));offset=count//2;signal=np.zeros(count);signal[offset:offset+len(flow)]=flow-flow[0];signal[offset+len(flow):]=flow[-1]-flow[0];F=np.fft.rfft(signal);Omega=2*np.pi*np.arange(len(F))/count
 H=np.zeros(len(F),complex);wsum=np.zeros(len(F));bound=np.inf
 for radius in range(min(20,nsub//2)):
  for alias in ([0] if radius==0 else [-radius,radius]):
   omega=(Omega+2*np.pi*alias)/nsub
   with np.errstate(divide='ignore',invalid='ignore'):weight=(np.sin(Omega/2)/(nsub*np.sin(omega/2)))**2
   weight[0]=1. if alias==0 else 0.
   H+=weight*transfer(frame,dt,omega);wsum+=weight
  remaining=np.maximum(0,1-wsum);closest=(2*np.pi*(radius+1)-Omega)/nsub
  env=transfer(frame,dt,closest,envelope=True)
  terms=abs(F)*remaining*env;bound=float((terms[0]+terms[-1]+2*terms[1:-1].sum())/count)
  if bound<=tolerance_cms:break
 else:raise ValueError('Alias remainder bound too large: '+str(bound))
 q=np.fft.irfft(F*H,n=count)[offset:offset+len(flow)]+flow[0]
 return q,dict(alias_radius=radius,omitted_alias_error_bound_cms=bound,fft_length=count,padding=padding)
