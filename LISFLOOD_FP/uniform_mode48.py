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
def evolve(h,q,dx,S,n,dt,steps,k):
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
            val=(q[i]-G*dt*hf*slope)/(1+G*dt*n*n*abs(q[i])/hf**(7/3))
            cap=hf*np.sqrt(G*hf)
            if abs(val)>cap:capped+=1
            qnew[i]=max(-cap,min(cap,val))
        for i in range(N):h[i]-=dt/dx*(qnew[i]-qnew[(i-1)%N])
        q=qnew
    return history,capped,float(min(h)),float(max(h))
def matrix(h,q,dx,S,n,dt,k):
    r=G*dt*n*n*q/h**(7/3);A=(1-r)/(1+r)
    C=G*dt*((10/3)*S+h/dx*(1-np.exp(1j*k)))/(1+r)
    D=dt/dx*(1-np.exp(-1j*k))
    return np.array([[1-D*C,-D*A],[C,A]])
if __name__=='__main__':
    out=Path(__file__).resolve().parent/'results/broad_15_100m/investigation/positivity48';n=.020799605931143154;records=[]
    # Representative uniform interiors, sampled from example 48 initial state.
    for reach,h,S in [(65,.848191594080612,.0034947845012664545),(161,1.0804578636021311,.0018810732578538286),(165,.7856739098117157,.0018102635742379334)]:
        q=h**(5/3)*np.sqrt(S)/n;Fr=q/(h*np.sqrt(G*h))
        for dx in [100.,50.,25.]:
            for dt in [3.,1.5,.75]:
                N=int(4000/dx);k=2*np.pi/N;M=matrix(h,q,dx,S,n,dt,k);ev,V=np.linalg.eig(M);idx=int(np.argmax(abs(ev)));lam=ev[idx];v=V[:,idx]/V[0,idx];eps=1e-7;phase=np.exp(1j*k*np.arange(N));hh=h+eps*np.real(phase*v[0]);qq=q+eps*np.real(phase*v[1]);steps=int(3600/dt)
                history,cap,hmin,hmax=evolve(hh.copy(),qq.copy(),dx,S,n,dt,steps,k);observed=history[-1,1]/history[0,1];predicted=abs(lam)**steps
                assert cap==0 and hmin>0
                assert abs(observed/predicted-1)<.001,(reach,dx,dt,observed,predicted)
                rec=dict(reach=reach,h=h,S=S,n=n,Fr=Fr,dx=dx,dt=dt,wavelength_m=4000.,hours=1.,observed_amplification=observed,linear_prediction=predicted,capped_face_steps=cap)
                records.append(rec);print(json.dumps(rec),flush=True)
    (out/'uniform_mode_results.json').write_text(json.dumps(records,indent=2))
