"""Small convex quadratic program with exhaustive active-set certificates."""
import itertools
import numpy as np


def quadratic_projection(C,target,weights,A,b):
    """Minimize sum((weights*(C*x-target))**2), subject to A*x >= b.

    x is scaled head displacement; up to three boundary candidates in this pilot.
    Enumerating independent active sets avoids a numerical optimizer's stop flag
    being mistaken for feasibility or optimality. Positive-definite objective only.
    """
    C=np.asarray(C,float);target=np.asarray(target,float);weights=np.asarray(weights,float)
    A=np.asarray(A,float);b=np.asarray(b,float);n=C.shape[1]
    w=weights/weights.max();Q=(C*w[:,None]).T@(C*w[:,None]);g=C.T@(w*w*target)
    if np.linalg.eigvalsh(Q).min()<=1e-14*np.linalg.eigvalsh(Q).max():raise ValueError('Exterior objective does not uniquely constrain boundary heads')
    best=None
    for k in range(n+1):
        for active in itertools.combinations(range(len(b)),k):
            rows=A[list(active)] if k else np.zeros((0,n));rhs=b[list(active)] if k else np.zeros(0)
            matrix=np.block([[Q,-rows.T],[rows,np.zeros((k,k))]])
            try:z=np.linalg.solve(matrix,np.r_[g,rhs])
            except np.linalg.LinAlgError:continue
            x=z[:n];lam=z[n:]
            if (A@x < b-1e-8).any() or (lam < -1e-8).any():continue
            objective=float(np.sum((w*(C@x-target))**2));stationarity=float(np.max(abs(Q@x-g-rows.T@lam)))
            if best is None or objective<best[0]:best=(objective,x,active,lam,stationarity)
    if best is None:raise ValueError('No certified feasible optimum')
    val,x,active,lam,stationarity=best
    return x,dict(normalized_objective=val,active_constraints=list(active),multipliers=lam.tolist(),stationarity=stationarity,primal_residual=max(0.,float(np.max(b-A@x))) if len(b) else 0.)
