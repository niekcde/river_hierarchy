"""Constrained Huber reference profiles with auditable disconnected-gap anchors.

This is an opt-in candidate method; it does not alter the existing profile API.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import factorized
from .reference_profile import _curvature_penalty


def weighted_median(values, weights):
    order = np.argsort(values)
    return float(np.interp(.5 * np.sum(weights), np.cumsum(weights[order]), values[order]))


def huber_loss(residual, delta):
    r = np.abs(residual)
    return np.where(r <= delta, r*r, 2*delta*r-delta*delta)


def disconnected_gap_anchors(profile, observations, minimum_slope):
    """Find spatial gaps between unions of acquisition spans, not every missing node.

    A pass spanning a missing-data interval connects the two sides and excludes
    that interval from this fallback. Static drop must satisfy the chosen floor;
    incompatible static references raise rather than being silently replaced.
    """
    spans = observations.groupby('acquisition_id').reference_section_dist_out_m.agg(['min', 'max']).sort_values('min')
    if spans.empty:
        raise ValueError('No acquisition spans')
    x = profile.reference_section_dist_out_m.to_numpy(float)
    static = profile.static_wse_m.to_numpy(float)
    end = float(spans.iloc[0]['max'])
    gaps = []
    for row in spans.itertuples():
        if row.min > end:
            left, right = end, float(row.min)
            drop = float(np.diff(np.interp([left, right], x, static))[0])
            required = minimum_slope * (right-left)
            if not np.isfinite(drop) or drop < required:
                raise ValueError(f'Static gap drop {drop} incompatible with floor drop {required}: {left}–{right}')
            gaps.append(dict(start_m=left, end_m=right, drop_m=drop,
                             floor_drop_m=required, source='static_SWORD_endpoint_difference'))
        end = max(end, float(row.max))
    return gaps


def _constraints(x, n_acq, floor, gaps, bridges):
    n = len(x); nv = n+n_acq
    D = sparse.hstack([sparse.diags([-np.ones(n-1), np.ones(n-1)], [0,1], shape=(n-1,n)), sparse.csr_matrix((n-1,n_acq))])
    gauge = sparse.csr_matrix((np.ones(n_acq)/n_acq, (np.zeros(n_acq), n+np.arange(n_acq))), shape=(1,nv))
    matrices = [D, gauge]; lower = list(floor*np.diff(x))+[0.]; upper = [np.inf]*(n-1)+[0.]
    linear_spans = []
    for g in gaps:
        il, ih = [int(np.argmin(abs(x-g[key]))) for key in ['start_m','end_m']]
        row = sparse.lil_matrix((1,nv)); row[0,ih]=1.; row[0,il]=-1.
        matrices.append(row.tocsr()); lower.append(g['drop_m']); upper.append(g['drop_m'])
        linear_spans.append((il, ih))
    for b in bridges:
        lo = b.get('resolved_start_distance_m', b.get('requested_start_distance_m'))
        hi = b.get('resolved_end_distance_m', b.get('requested_end_distance_m'))
        linear_spans.append((int(np.argmin(abs(x-lo))), int(np.argmin(abs(x-hi)))))
    # Bridges are constraints inside the fit, not post-fit changes to the optimum.
    for il, ih in linear_spans:
        for j in range(il+1, ih):
            t = (x[j]-x[il])/(x[ih]-x[il])
            row = sparse.lil_matrix((1,nv)); row[0,j]=1.; row[0,il]=-(1-t); row[0,ih]=-t
            matrices.append(row.tocsr()); lower.append(0.); upper.append(0.)
    return sparse.vstack(matrices).tocsr(), np.array(lower), np.array(upper)


def prepare_problem(profile, observations, reference_slope, *, floor_fraction=.05,
                    weight_total=None, gaps=(), bridges=(), initial_heights=None):
    x = profile.reference_section_dist_out_m.to_numpy(float)
    o = observations.reset_index(drop=True)
    if len(x)<2 or np.any(np.diff(x)<=0) or len(o)==0:
        raise ValueError('Need increasing grid and nonempty observations')
    acqs = pd.Index(sorted(o.acquisition_id.unique())); ai = acqs.get_indexer(o.acquisition_id)
    ni = np.searchsorted(x, o.reference_section_dist_out_m)
    if np.any(ni>=len(x)) or not np.allclose(x[ni], o.reference_section_dist_out_m, atol=1e-6, rtol=0):
        raise ValueError('Observations must lie on profile grid')
    n=len(x); N=len(o); nv=n+len(acqs)
    M=sparse.coo_matrix((np.ones(2*N),(np.r_[np.arange(N),np.arange(N)],np.r_[ni,n+ai])),shape=(N,nv)).tocsr()
    weights=(N if weight_total is None else weight_total)/(o.cycle.nunique()*o.groupby('cycle').wse_m.transform('size').to_numpy())
    seed = profile.static_wse_m.to_numpy(float) if initial_heights is None else np.asarray(initial_heights,float)
    center=float(np.median(seed)); y=o.wse_m.to_numpy(float)-center
    offsets=pd.Series(y-(seed-center)[ni]).groupby(ai).median().reindex(range(len(acqs))).to_numpy(); common=offsets.mean()
    initial=np.r_[seed-center+common,offsets-common]
    A,lower,upper=_constraints(x,len(acqs),floor_fraction*reference_slope,gaps,bridges)
    penalty=sparse.block_diag((_curvature_penalty(x),sparse.csr_matrix((len(acqs),len(acqs)))),format='csc')
    return dict(x=x,o=o,acqs=acqs,ni=ni,ai=ai,M=M,A=A,lower=lower,upper=upper,
                weights=weights,center=center,y=y,initial=initial,penalty=penalty)


def fit_robust_profile(problem, lambda_value, delta, *, initial=None, max_iterations=100000,
                       primal_tolerance=2e-6, dual_tolerance=2e-5):
    """Solve convex Huber + curvature objective using scaled ADMM.

    Residual and constraint tolerances are absolute, in centred metre coordinates.
    The observation prox has an analytic Huber solution. No IRLS scale updates.
    delta=inf provides the squared-loss anchor for training-only MAD estimation.
    """
    if lambda_value<=0 or delta<=0:
        raise ValueError('Positive lambda and Huber delta required')
    start=time.monotonic(); M=problem['M']; A=problem['A']; B=sparse.vstack([M,A]).tocsc(); Bt=B.T.tocsc()
    N=len(problem['y']); w=problem['weights']; y=problem['y']; lo=problem['lower']; hi=problem['upper']
    P=(2*lambda_value*problem['penalty']).tocsc(); n=P.shape[0]
    v=problem['initial'].copy() if initial is None else initial.copy(); rho=1.; sigma=1e-8
    gram=(Bt@B).tocsc(); identity=sparse.eye(n,format='csc')
    def factor():return factorized(P+rho*gram+sigma*identity)
    solve=factor(); z=B@v; z[N:]=np.clip(z[N:],lo,hi); u=np.zeros(len(z))
    for it in range(max_iterations):
        v=solve(rho*(Bt@(z-u))+sigma*v); bv=B@v; previous=z.copy(); q=bv+u
        t=q[:N]-y; shrink=2*w/rho
        if np.isinf(delta):res=t/(1+shrink)
        else:res=np.where(abs(t)<=delta*(1+shrink),t/(1+shrink),t-np.sign(t)*shrink*delta)
        z[:N]=y+res; z[N:]=np.clip(q[N:],lo,hi); u+=bv-z
        primal=float(np.max(abs(bv-z))); dual=float(np.max(abs(rho*(Bt@(z-previous)))))
        if primal<=primal_tolerance and dual<=dual_tolerance:break
        if (it+1)%100==0:
            ratio=(primal/primal_tolerance)/(dual/dual_tolerance+1e-30)
            next_rho=min(1e5,rho*2) if ratio>10 else max(1e-5,rho/2) if ratio<.1 else rho
            if next_rho!=rho:u*=rho/next_rho;rho=next_rho;solve=factor()
    else:raise RuntimeError(f'ADMM not converged: primal {primal}, dual {dual}')
    residual=y-M@v; constraint=A@v; violation=float(max(0,np.max(lo-constraint),np.max(constraint-hi)))
    loss=residual**2 if np.isinf(delta) else huber_loss(residual,delta)
    audit=dict(iterations=it+1,primal_residual_m=primal,dual_residual=dual,constraint_violation_m=violation,
               objective=float(w@loss+.5*v@(P@v)),seconds=time.monotonic()-start,converged=True)
    return v[:len(problem['x'])]+problem['center'],v,audit


def training_delta(problem, squared_solution):
    residual=problem['y']-problem['M']@squared_solution;w=problem['weights']
    scale=max(1e-6,1.4826*weighted_median(abs(residual-weighted_median(residual,w)),w))
    return 1.345*scale


def score_held_cycles(observations, x, heights, delta):
    """Stage offsets optimized independently on held acquisitions; score shape only."""
    o=observations.copy(); r=o.wse_m.to_numpy()-np.interp(o.reference_section_dist_out_m,x,heights)
    o['raw']=r; o['robust']=0.;o['squared']=0.
    for _,g in o.groupby('acquisition_id'):
        v=g.raw.to_numpy(); a,b=float(v.min()),float(v.max())
        location=a if a==b else brentq(lambda m:float(np.clip(v-m,-delta,delta).sum()),a,b)
        o.loc[g.index,'robust']=huber_loss(v-location,delta)
        o.loc[g.index,'squared']=(v-v.mean())**2
    return o.groupby('cycle')[['robust','squared']].mean()
