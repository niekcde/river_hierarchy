"""Controlled periodic-slope test of the supplied local-inertial update.
No junctions, width transitions, external forcing changes or outlet boundary.
The bed drop is handled across the periodic seam; only perturbations are periodic.
Analytic linearization is specific to uniform positive subcritical flow, with
upstream surface higher than downstream and the critical cap inactive.
"""
from pathlib import Path
import json
import numpy as np
from numba import njit
G=9.80665
@njit(cache=True)
def evolve(h,q,dx,S,n,dt,steps,k,theta=1.):
    N=len(h);history=np.zeros((steps+1,3));capped=0
    for s in range(steps+1):
        ah=0j;aq=0j
        for i in range(N):
            phase=np.exp(-1j*k*i);ah+=h[i]*phase;aq+=q[i]*phase
        history[s]=np.array([s*dt,abs(ah)*2/N,abs(aq)*2/N])
        if s==steps:break
        qnew=np.empty(N)
        for i in range(N):
            j=(i+1)%N;hf=max(h[i],h[j]-S*dx,1e-12)
            slope=-S+(h[j]-h[i])/dx
            val=(theta*q[i]+.5*(1-theta)*(q[(i-1)%N]+q[(i+1)%N])-G*dt*hf*slope)/(1+G*dt*n*n*abs(q[i])/hf**(7/3))
            cap=hf*np.sqrt(G*hf)
            if abs(val)>cap:capped+=1
            qnew[i]=max(-cap,min(cap,val))
        for i in range(N):h[i]-=dt/dx*(qnew[i]-qnew[(i-1)%N])
        q=qnew
    return history,capped,float(min(h)),float(max(h))
def matrix(h,q,dx,S,n,dt,k,theta=1.):
    r=G*dt*n*n*q/h**(7/3);A=(theta+(1-theta)*np.cos(k)-r)/(1+r)
    C=G*dt*((10/3)*S+h/dx*(1-np.exp(1j*k)))/(1+r)
    D=dt/dx*(1-np.exp(-1j*k))
    return np.array([[1-D*C,-D*A],[C,A]])

if __name__=='__main__':
    out=Path(__file__).resolve().parent/'results/broad_15_100m/investigation/stabilized48';out.mkdir(exist_ok=True);n=.020799605931143154;records=[]
    for reach,h,S in [(65,.848191594080612,.0034947845012664545),(161,1.0804578636021311,.0018810732578538286)]:
      q=h**(5/3)*np.sqrt(S)/n
      for dx in [100.,50.,25.]:
       for dt in [3.,1.5,.75]:
        for theta in [1.,.9,.8,.7]:
         N=int(4000/dx);k=2*np.pi/N;M=matrix(h,q,dx,S,n,dt,k,theta);ev,V=np.linalg.eig(M);idx=int(np.argmax(abs(ev)));lam=ev[idx];v=V[:,idx]/V[0,idx];eps=1e-7;phase=np.exp(1j*k*np.arange(N));hh=h+eps*np.real(phase);qq=q+eps*np.real(phase*v[1]);steps=int(3600/dt)
         hist,cap,hmin,hmax=evolve(hh,qq,dx,S,n,dt,steps,k,theta);observed=hist[-1,1]/hist[0,1];predicted=abs(lam)**steps
         assert cap==0 and hmin>0 and abs(observed/predicted-1)<.001
         records.append(dict(reach=reach,dx=dx,dt=dt,theta=theta,observed_amplification=observed,linear_prediction=predicted))
    (out/'uniform_results.json').write_text(json.dumps(records,indent=2));print('PASS',len(records),'linearized vs nonlinear uniform mode tests')
