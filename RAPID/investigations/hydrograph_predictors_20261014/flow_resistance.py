"""Unit-flow convex solve of the same signed quadratic resistance law.

Independent numerical check of the existing head-form solver. Primal/dual
energy bounds certify equivalent K; no routed quantities enter this solve.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_unit_flow(edges,source,sink,tolerance=1e-8,max_iterations=250):
    nodes=sorted(set(edges.u)|set(edges.v));interior=[n for n in nodes if n!=sink];index={n:i for i,n in enumerate(interior)}
    rr=[];cc=[];vv=[]
    for j,(u,v) in enumerate(zip(edges.u,edges.v)):
        if u!=sink:rr.append(index[u]);cc.append(j);vv.append(1.)
        if v!=sink:rr.append(index[v]);cc.append(j);vv.append(-1.)
    B=sparse.csc_matrix((vv,(rr,cc)),shape=(len(interior),len(edges)))
    k=edges.K_e.to_numpy(float);scale=float(np.median(k));k=k/scale
    b=np.zeros(len(interior));b[index[source]]=1.
    def laplace(w,rhs):
        L=(B@sparse.diags(w)@B.T).tocsc()
        # Diagonal equilibration improves solves spanning large channel-width ratios.
        diag=1/np.sqrt(L.diagonal());D=sparse.diags(diag)
        return diag*spsolve((D@L@D).tocsc(),diag*rhs)
    w=1/np.sqrt(k);h=laplace(w,b);q=w*(B.T@h)
    best_lower=0.;best_q=q.copy();best_upper=np.inf;gap=np.inf
    for iteration in range(max_iterations):
        energy=np.sum(k*np.abs(q)**3)
        g=k*q*np.abs(q)
        floor=max(np.max(np.abs(q))*1e-13,1e-15)
        inverse=1/(2*k*np.maximum(np.abs(q),floor))
        correction=b-B@q
        h=laplace(inverse,B@(inverse*g)+correction)
        dh=B.T@h
        bh=float(b@h);S=float(np.sum(np.abs(dh)**1.5/np.sqrt(k)))
        lower=bh**3/S**2 if bh>0 and S>0 else 0.
        continuity=float(np.max(np.abs(B@q-b)))
        # The upper bound requires a feasible flow. Keep only numerically feasible iterates.
        if continuity<1e-8 and energy<best_upper:best_upper=energy;best_q=q.copy()
        best_lower=max(best_lower,lower)
        gap=abs(best_upper-best_lower)/best_upper if np.isfinite(best_upper) else np.inf
        if gap<=tolerance and continuity<1e-8:
            return dict(K_eq=best_upper*scale,lower_K=best_lower*scale,relative_gap=max(0.,gap),
                        continuity=continuity,success=True,iterations=iteration+1,flows=best_q)
        step=inverse*(dh-g);slope=float(g@step)
        length=1.
        for _ in range(60):
            trial=q+length*step
            if np.sum(k*np.abs(trial)**3)/3 <= energy/3+1e-4*length*min(slope,0.)+1e-15*max(energy,1.):break
            length*=.5
        q=trial
    return dict(K_eq=best_upper*scale,lower_K=best_lower*scale,relative_gap=gap,
                continuity=float(np.max(abs(B@best_q-b))),success=False,iterations=max_iterations,flows=best_q)
