"""EXPERIMENTAL q-centred de Almeida-style update plus donor-volume limiter.
Not production. Smoothing restricted to regular interiors of physical reaches;
physical junctions and boundaries retain the original update. This restriction
is our network adaptation, not a published general junction discretization.

Based on inertial_verification_kernel.py. Outgoing faces share a donor-cell
scaling factor computed from pre-step storage plus external positive inflow.
Incoming internal flux is not spent until the next step. Each face is changed
once, with identical flux used by both adjacent cells. Momentum history uses
the resulting limited flux. This ensures positivity, not hydraulic accuracy.

Audited array kernel for the 56 verification study (CPU, not MATLAB).

Normal flow uses the terminal-cell depth and terminal physical reach slope:
Q=B*h**(5/3)*sqrt(S)/n. This is a zero depth-gradient outlet closure.
Fixed-stage tests use a boundary face at half the terminal-cell length.
"""
import sys
import numpy as np
from numba import njit

G=9.80665

@njit(cache=True)
def donor_limit(h,area,a,b,wf,q,source,inlet,dt):
    N=len(h);outgoing=np.zeros(N);factor=np.ones(N)
    for j in range(len(a)):
        Q=q[j]*wf[j]
        if Q>0:outgoing[a[j]]+=Q
        elif b[j]<N:outgoing[b[j]]-=Q
    if inlet<0:raise ValueError('Negative external source unsupported by this diagnostic')
    for i in range(N):
        available=max(h[i],0.)*area[i]
        if i==source:available+=dt*inlet
        if dt*outgoing[i]>available:
            factor[i]=(1.-1e-12)*available/(dt*outgoing[i])
    count=0;deferred=0.;smallest=1.
    for j in range(len(a)):
        donor=a[j] if q[j]>=0 else b[j]
        if donor<N and factor[donor]<1.:
            old=q[j];q[j]*=factor[donor]
            count+=1;deferred+=abs(old-q[j])*wf[j]*dt
            smallest=min(smallest,factor[donor])
    return count,deferred,smallest

@njit(cache=True)
def integrate(z,w,dx,a,b,wf,dxf,source,n,outlet_slope,stage,normal,times,forcing,h,q,cfl,max_dt,sample_indices,monitor_face=-1,abort_on_invalid=False,theta=1.,smooth_left=None,smooth_right=None):
    N=len(w); M=len(a); area=w*dx
    h=h.copy();q=q.copy()
    if theta<0. or theta>1.:raise ValueError("theta outside [0,1]")
    if smooth_left is None:
        if theta<1.:raise ValueError("Missing smoothing stencil")
        smooth_left=np.full(M,-1,dtype=np.int64)
    if smooth_right is None:
        if theta<1.:raise ValueError("Missing smoothing stencil")
        smooth_right=np.full(M,-1,dtype=np.int64)
    rows=np.zeros((len(times),9));snap=np.zeros((len(sample_indices),N))
    qsnap=np.zeros((len(sample_indices),M));hmax=h.copy();hmin=h.copy()
    positive_count=0;deferred_volume=0.;minimum_factor=1.;first_positive_time=-1.
    initial=np.sum(area*h);vin=0.;vout=0.;clip=0.;steps=0;limited=0;monitor_volume=0.
    min_dt=1e99;max_used=0.;minQ=1e99;maxFr=0.;resmax=0.
    balance=np.zeros(N);eta=np.zeros(N);hf=np.zeros(M)
    for it in range(len(times)):
        if it>0:
            t=times[it-1];stop=times[it];inlet=forcing[it-1]
            while t<stop-1e-8:
                dt=min(max_dt,stop-t)
                for i in range(N):
                    dt=min(dt,cfl*dx[i]/np.sqrt(G*max(h[i],1e-12)))
                    eta[i]=z[i]+h[i];balance[i]=0.
                qblend=q.copy()
                if theta<1.:
                    for j in range(M):
                        if smooth_left[j]>=0 and smooth_right[j]>=0:
                            qblend[j]=theta*q[j]+.5*(1-theta)*(q[smooth_left[j]]+q[smooth_right[j]])
                for j in range(M):
                    i=a[j];k=b[j]
                    if k==N and normal:
                        hf[j]=max(h[i],1e-12)
                        newq=max(h[i],0.)**(5./3.)*np.sqrt(outlet_slope)/n
                        # Boundary Manning relation is not limited; diagnose Fr.
                    else:
                        etab=stage if k==N else eta[k]
                        zb=z[N] if k==N else z[k]
                        hf[j]=max(max(eta[i],etab)-max(z[i],zb),1e-12)
                        slope=(etab-eta[i])/dxf[j]
                        newq=(qblend[j]-G*dt*hf[j]*slope)/(1+G*dt*n*n*abs(q[j])/hf[j]**(7./3.))
                        limit=hf[j]*np.sqrt(G*hf[j])
                        if abs(newq)>limit:limited+=1
                        newq=max(-limit,min(limit,newq))
                    q[j]=newq
                count,deferred,smallest=donor_limit(h,area,a,b,wf,q,source,inlet,dt)
                if count>0 and first_positive_time<0:first_positive_time=t
                positive_count+=count;deferred_volume+=deferred;minimum_factor=min(minimum_factor,smallest)
                for j in range(M):
                    i=a[j];k=b[j];newq=q[j];Q=newq*wf[j]
                    if j==monitor_face:monitor_volume+=Q*dt
                    balance[i]-=Q
                    if k<N:balance[k]+=Q
                    else:vout+=Q*dt
                    minQ=min(minQ,Q);maxFr=max(maxFr,abs(newq)/(hf[j]*np.sqrt(G*hf[j])))
                balance[source]+=inlet
                for i in range(N):
                    trial=h[i]+dt*balance[i]/area[i]
                    if trial<0:clip-=trial*area[i]
                    h[i]=max(0.,trial);hmax[i]=max(hmax[i],h[i]);hmin[i]=min(hmin[i],h[i])
                vin+=inlet*dt;t+=dt;steps+=1;min_dt=min(min_dt,dt);max_used=max(max_used,dt)
        storage=np.sum(area*h);out=0.
        for j in range(M):
            if b[j]==N:out+=q[j]*wf[j]
        residual=storage-initial-vin+vout-clip;resmax=max(resmax,abs(residual))
        if abort_on_invalid and (clip>1e-6 or limited>0 or not np.isfinite(storage)):
            print('REJECTED_EVENT time_s clipping_m3 limited_face_steps:',times[it],clip,limited)
            if not np.isfinite(storage):raise ValueError('Nonfinite storage')
            if clip>1e-6:raise ValueError('Depth clipping in event')
            raise ValueError('Critical-flow limiter active in event')
        monitor_q=q[monitor_face]*wf[monitor_face] if monitor_face>=0 else out
        rows[it,:]=np.array([times[it],out,storage,vin,vout,clip,residual,monitor_q,monitor_volume if monitor_face>=0 else vout])
        for s in range(len(sample_indices)):
            if sample_indices[s]==it:snap[s,:]=h;qsnap[s,:]=q*wf
    balance[:]=0.
    for j in range(M):
        balance[a[j]]-=q[j]*wf[j]
        if b[j]<N:balance[b[j]]+=q[j]*wf[j]
    balance[source]+=forcing[-1]
    audit=np.array([steps,limited,clip,resmax,np.max(hmax),np.min(hmin),minQ,min_dt,max_used,np.max(np.abs(balance)),maxFr,positive_count,deferred_volume,minimum_factor,first_positive_time])
    return rows,snap,qsnap,h,q,hmax,hmin,audit

if __name__=='__main__':
    data={k:(v.item() if v.ndim==0 else v) for k,v in dict(np.load(sys.argv[1])).items()};result=integrate(**data)
    np.savez_compressed(sys.argv[2],**dict(zip(['wave','depth','flow','h','q','hmax','hmin','audit'],result)))
