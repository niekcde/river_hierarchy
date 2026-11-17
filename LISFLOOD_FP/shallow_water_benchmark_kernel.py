"""Independent 1D finite-volume benchmark on flat/uniformly sloping channels.
Full SWE: U=(h,q), F=(q,q^2/h+g*h^2/2), source=(0,g*h*S-g*n^2*q*abs(q)/h^(7/3)).
MUSCL reconstruction with generalized minmod, HLL flux, SSP-RK2.
Switch advective=False removes q^2/h, retaining otherwise identical numerics.
Only wet states, constant width, constant slope. No dry-bed/junction/general
bed treatment: this is NOT a production network solver. Periodic perturbations
on a sloping bed use a constant gravitational source, not a discontinuous bed seam.
"""
import numpy as np
from numba import njit
G=9.80665
@njit(cache=True)
def mm(a,b,c):
    if a>0 and b>0 and c>0:return min(a,b,c)
    if a<0 and b<0 and c<0:return max(a,b,c)
    return 0.
@njit(cache=True)
def rhs(U,dx,S,n,advective,periodic,limited=True):
    N=len(U);slopes=np.zeros_like(U);flux=np.zeros((N+1,2))
    for i in range(N):
        l=(i-1)%N if periodic else max(i-1,0);r=(i+1)%N if periodic else min(i+1,N-1)
        for c in range(2):
            slopes[i,c]=mm(1.5*(U[i,c]-U[l,c]),.5*(U[r,c]-U[l,c]),1.5*(U[r,c]-U[i,c])) if limited else .5*(U[r,c]-U[l,c])
    for j in range(N+1):
        l=(j-1)%N if periodic else max(j-1,0);r=j%N if periodic else min(j,N-1)
        hl=U[l,0]+.5*slopes[l,0];ql=U[l,1]+.5*slopes[l,1];hr=U[r,0]-.5*slopes[r,0];qr=U[r,1]-.5*slopes[r,1]
        if not periodic and j==0:hl=hr;ql=qr
        if not periodic and j==N:hr=hl;qr=ql
        if min(hl,hr)<=0:raise ValueError('Nonpositive reconstructed depth in wet-only benchmark')
        ul=ql/hl;ur=qr/hr;cl=np.sqrt(G*hl);cr=np.sqrt(G*hr)
        if advective:
            sl=min(ul-cl,ur-cr);sr=max(ul+cl,ur+cr);fl=ql*ul+.5*G*hl*hl;fr=qr*ur+.5*G*hr*hr
        else:
            sl=-max(cl,cr);sr=max(cl,cr);fl=.5*G*hl*hl;fr=.5*G*hr*hr
        if sl>=0:flux[j,0]=ql;flux[j,1]=fl
        elif sr<=0:flux[j,0]=qr;flux[j,1]=fr
        else:
            flux[j,0]=(sr*ql-sl*qr+sl*sr*(hr-hl))/(sr-sl)
            flux[j,1]=(sr*fl-sl*fr+sl*sr*(qr-ql))/(sr-sl)
    out=np.empty_like(U)
    for i in range(N):
        out[i,0]=-(flux[i+1,0]-flux[i,0])/dx
        h=U[i,0];q=U[i,1]
        out[i,1]=-(flux[i+1,1]-flux[i,1])/dx+G*h*S-G*n*n*q*abs(q)/h**(7/3)
    return out,flux[0,0]-flux[N,0]
@njit(cache=True)
def integrate(U,dx,S,n,T,cfl=.35,advective=True,periodic=True,max_dt=1e99,wavenumber=0.,limited=True):
    U=U.copy();N=len(U);t=0.;steps=0;initial=np.sum(U[:,0])*dx;netboundary=0.;hmin=np.min(U[:,0]);hmax=np.max(U[:,0]);maxres=0.
    history=np.zeros((361,5));sample=0
    while sample<361:
        target=T*sample/360
        while t<target-1e-10:
            speed=0.;damping=0.
            for i in range(N):
                h=U[i,0];q=U[i,1];speed=max(speed,np.sqrt(G*h)+(abs(q/h) if advective else 0.));damping=max(damping,2*G*n*n*abs(q)/h**(7/3))
            dt=min(target-t,max_dt,cfl*dx/speed,.5/max(damping,1e-12))
            R,fb=rhs(U,dx,S,n,advective,periodic,limited);V=U+dt*R
            if np.min(V[:,0])<=0:raise ValueError('Nonpositive RK stage')
            R2,fb2=rhs(V,dx,S,n,advective,periodic,limited);U=.5*(U+V+dt*R2)
            if np.min(U[:,0])<=0 or not np.isfinite(U).all():raise ValueError('Invalid RK update')
            netboundary+=.5*dt*(fb+fb2);t+=dt;steps+=1;hmin=min(hmin,np.min(U[:,0]));hmax=max(hmax,np.max(U[:,0]));maxres=max(maxres,abs(np.sum(U[:,0])*dx-initial-netboundary))
        ah=0j
        for i in range(N):ah+=U[i,0]*np.exp(-1j*wavenumber*(i+.5)*dx)
        history[sample,:]=np.array([t,2*abs(ah)/N,2*ah.real/N,2*ah.imag/N,np.sum(U[:,0])*dx]);sample+=1
    return U,history,np.array([steps,hmin,hmax,maxres,netboundary])
