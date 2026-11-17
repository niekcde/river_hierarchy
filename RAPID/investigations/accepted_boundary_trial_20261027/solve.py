"""Convex head fitting with explicit feasibility and KKT residual checks."""
import numpy as np
from scipy.optimize import linprog,minimize,nnls

def fit_internal(edges,fixed,reference_slope,floor_drops):
 nodes=set(edges.u)|set(edges.v);free=sorted(nodes-set(fixed));ix={v:i for i,v in enumerate(free)}
 if not free:
  drops=np.array([fixed[r.u]-fixed[r.v] for r in edges.itertuples()]);assert np.min(drops-floor_drops)>=-1e-8
  return dict(fixed),dict(free_nodes=0,primal_residual_m=max(0.,float(np.max(floor_drops-drops))),stationarity=0.)
 # Represent free heads about a fixed-head offset; scaling makes a typical edge target drop order one.
 offset=float(np.mean(list(fixed.values())));scale=max(float(np.median(edges.link_length_m)*reference_slope),.001)
 C=np.zeros((len(edges),len(free)));base=np.zeros(len(edges))
 for j,r in enumerate(edges.itertuples()):
  for v,sgn in [(r.u,1),(r.v,-1)]:
   if v in ix:C[j,ix[v]]+=sgn*scale;base[j]+=sgn*offset
   else:base[j]+=sgn*fixed[v]
 A=C/scale;b=(floor_drops-base)/scale;active=np.any(A!=0,axis=1)
 if np.any(b[~active]>1e-8/scale):raise ValueError('Fixed edge violates floor in internal fit')
 A=A[active];b=b[active]
 target=edges.link_length_m.to_numpy()*reference_slope
 M=C/target[:,None];y=1-base/target
 # Equal-per-edge relative slope objective; common normalisation does not change optimum.
 norm=max(float(np.linalg.norm(M,2)**2),1.);Q=M.T@M/norm;g=M.T@y/norm
 uncon=np.linalg.lstsq(M,y,rcond=None)[0]
 if np.all(A@uncon>=b-1e-10):x=uncon;message='Unconstrained optimum feasible'
 else:
  feasible=linprog(np.zeros(len(free)),A_ub=-A,b_ub=-b,bounds=[(None,None)]*len(free),method='highs',options={'primal_feasibility_tolerance':1e-9})
  if not feasible.success:raise ValueError('Internal LP infeasible: '+feasible.message)
  x,message=active_set_qp(Q,g,A,b,feasible.x)

 residual=max(0.,float(np.max(b-A@x))*scale);tight=np.where(A@x-b<1e-7)[0];grad=Q@x-g
 lam=nnls(A[tight].T,grad,maxiter=max(100,10*len(tight)))[0] if len(tight) else np.zeros(0)
 station=float(np.max(abs(grad-A[tight].T@lam))) if len(tight) else float(np.max(abs(grad)))
 if residual>1e-8 or station>1e-5:raise ValueError(f'Internal QP certificate failed: primal={residual}, stationarity={station}')
 heads=dict(fixed);heads.update({v:float(offset+scale*x[i]) for v,i in ix.items()})
 return heads,dict(free_nodes=len(free),primal_residual_m=residual,stationarity=station,objective=float(np.sum((M@x-y)**2)),active_constraints=len(tight),message=message)

def fit_primary(C,weights,A,b):
 """Relative-drop QP, then minimum boundary displacement on its optimal face.

 The tie-break is only used if the exterior objective is rank deficient. It
 never trades increased exterior disturbance for smaller boundary movement.
 """
 import itertools
 from solver import quadratic_projection
 C=np.asarray(C,float);weights=np.asarray(weights,float);A=np.asarray(A,float);b=np.asarray(b,float);n=C.shape[1]
 w=weights/weights.max();Q=(C*w[:,None]).T@(C*w[:,None]);best=None
 for k in range(n+1):
  for active in itertools.combinations(range(len(b)),k):
   E=A[list(active)] if k else np.zeros((0,n));rhs=b[list(active)] if k else np.zeros(0);K=np.block([[Q,-E.T],[E,np.zeros((k,k))]]);v=np.r_[np.zeros(n),rhs]
   z=np.linalg.lstsq(K,v,rcond=1e-12)[0]
   if np.max(abs(K@z-v),initial=0)>1e-8:continue
   x=z[:n];lam=z[n:]
   if np.any(A@x<b-1e-8) or np.any(lam<-1e-8):continue
   score=float(x@Q@x)
   if best is None or score<best[0]-1e-12:best=(score,x,list(active),lam.tolist())
 if best is None:raise ValueError('No certified primary optimum')
 val,x,act,lam=best;tie=bool(np.linalg.matrix_rank(C)<n)
 if tie:
  # Every primary minimiser has the same C*x. Preserve it exactly in a
  # separate strictly convex minimum-displacement problem.
  target=C@x;At=np.vstack([A,C,-C]);bt=np.r_[b,target,-target]
  x,tqa=quadratic_projection(np.eye(n),np.zeros(n),np.ones(n),At,bt)
  assert np.max(abs(C@x-target),initial=0)<1e-7
 else:tqa=None
 return x,dict(primary_objective_normalized=float(np.sum((w*(C@x))**2)),primary_primal_residual=float(max(0.,np.max(b-A@x))),rank_deficient=tie,tie_break='minimum squared boundary displacement on the primary optimal face' if tie else None,tie_certificate=tqa,certified_primary_active_set=act,primary_multipliers=lam)


def active_set_qp(Q,g,A,b,x):
 """Primal feasible active-set QP; handles a zero-width feasible face."""
 x=x.copy();n=len(x);W=[]
 def independent(rows,j):
  return np.linalg.matrix_rank(A[rows+[j]],tol=1e-10)>len(rows)
 for j in np.where(A@x-b<1e-8)[0]:
  if independent(W,int(j)):W.append(int(j))
 for it in range(5000):
  E=A[W] if W else np.zeros((0,n));grad=Q@x-g;K=np.block([[Q,-E.T],[E,np.zeros((len(W),len(W)))]])
  rhs=np.r_[-grad,np.zeros(len(W))]
  try:z=np.linalg.solve(K,rhs)
  except np.linalg.LinAlgError:z=np.linalg.lstsq(K,rhs,rcond=1e-13)[0]
  direction=z[:n];lam=z[n:]
  if np.max(abs(direction),initial=0)<1e-9:
   if np.min(lam,initial=0)>=-1e-9:return x,f'Certified active-set optimum in {it+1} iterations'
   del W[int(np.argmin(lam))];continue
  alpha=1.;block=None;slack=A@x-b;ad=A@direction
  for j in range(len(b)):
   if j not in W and ad[j]<-1e-11:
    candidate=max(0.,slack[j])/(-ad[j])
    if candidate<alpha:alpha=candidate;block=j
  x+=alpha*direction
  if block is not None and independent(W,block):W.append(block)
  if np.min(A@x-b,initial=0)<-1e-7:raise ValueError('Active-set solver lost feasibility')
 raise ValueError('Active-set QP iteration limit')
