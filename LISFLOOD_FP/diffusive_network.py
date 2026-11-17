"""Experimental, wet-channel 1D diffusive-wave graph solver (not LISFLOOD-FP).

Rectangular sections, fixed bed, Manning friction, no inertia or floodplain.
Node-centred finite volumes: every segment contributes W*dx/2 storage area
to each endpoint. Junctions share stage and store only their incident half cells.
Signed face fluxes are calculated from stage gradients; no prescribed splits.
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, lil_matrix


def conveyance(depth, width, roughness):
    h = np.maximum(depth, 0.)  # evaluates trial states; accepted dry states rejected
    area = width * h
    radius = area / (width + 2 * h)
    return area * radius ** (2 / 3) / roughness


def normal_depth(discharge, width, slope, roughness):
    if min(discharge, width, slope, roughness) <= 0:
        raise ValueError('Normal-depth arguments must be positive')
    hi = 1.
    while conveyance(hi, width, roughness) * np.sqrt(slope) < discharge:
        hi *= 2
    return brentq(lambda h: conveyance(h, width, roughness) * np.sqrt(slope) - discharge, 1e-10, hi)


@dataclass
class Network:
    bed: np.ndarray
    storage_area: np.ndarray
    faces: pd.DataFrame
    source: int
    sink: int
    outlet_width: float
    outlet_slope: float
    roughness: float
    node_ids: list
    slope_epsilon: float = 1e-10

    @classmethod
    def from_frame(cls, frame, dx=200., roughness=.035):
        required = ['u', 'v', 'link_length_m', 'width', 'head_us', 'head_ds']
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise ValueError('Nonfinite geometry')
        if dx <= 0 or roughness <= 0 or (frame[['width','link_length_m']] <= 0).any().any():
            raise ValueError('Nonpositive dimensions or roughness')
        g = nx.MultiDiGraph()
        heads = {}
        for r in frame.itertuples():
            g.add_edge(int(r.u), int(r.v))
            for node, z in [(int(r.u), r.head_us), (int(r.v), r.head_ds)]:
                if node in heads and not np.isclose(heads[node], z, atol=1e-7, rtol=0):
                    raise ValueError(f'Inconsistent bed at junction {node}')
                heads[node] = float(z)
        roots = [n for n in g if g.in_degree(n) == 0]
        sinks = [n for n in g if g.out_degree(n) == 0]
        if not nx.is_directed_acyclic_graph(g) or len(roots) != 1 or len(sinks) != 1:
            raise ValueError('Input must be a connected one-inlet, one-outlet DAG')
        ids = sorted(heads)
        lookup = {n:i for i,n in enumerate(ids)}
        bed = [heads[n] for n in ids]
        faces = []
        for link, r in enumerate(frame.itertuples()):
            count = max(1, int(np.ceil(r.link_length_m / dx)))
            previous = lookup[int(r.u)]
            for j in range(count):
                if j == count-1:
                    end = lookup[int(r.v)]
                else:
                    end = len(bed)
                    bed.append(r.head_us + (r.head_ds-r.head_us)*(j+1)/count)
                    ids.append(f'link{link}:{j+1}')
                faces.append((previous,end,r.link_length_m/count,r.width,link,j,count,int(r.u),int(r.v)))
                previous = end
        f = pd.DataFrame(faces, columns=['a','b','dx','width','link','section','count','original_u','original_v'])
        area = np.bincount(f.a, weights=f.width*f.dx/2, minlength=len(bed))
        area += np.bincount(f.b, weights=f.width*f.dx/2, minlength=len(bed))
        terminal = frame[frame.v == sinks[0]]
        if len(terminal) != 1:
            raise ValueError('Prototype requires a single terminal reach for normal-flow outlet')
        r = terminal.iloc[0]
        slope = (r.head_us-r.head_ds)/r.link_length_m
        if slope <= 0:
            raise ValueError('Normal-flow outlet needs positive terminal bed slope')
        return cls(np.asarray(bed), area, f, lookup[roots[0]], lookup[sinks[0]],
                   float(r.width), float(slope), roughness, ids)

    def __post_init__(self):
        self.a = self.faces.a.to_numpy(int)
        self.b = self.faces.b.to_numpy(int)
        self.dx = self.faces.dx.to_numpy(float)
        self.width = self.faces.width.to_numpy(float)
        n, m = len(self.bed), len(self.faces)
        self.incidence = coo_matrix((np.r_[-np.ones(m),np.ones(m)],
            (np.r_[self.a,self.b],np.r_[np.arange(m),np.arange(m)])), shape=(n,m)).tocsr()
        pattern = lil_matrix((n+2,n+2), dtype=int)
        for a,b in zip(self.a,self.b):
            pattern[a,a] = pattern[a,b] = pattern[b,a] = pattern[b,b] = 1
        pattern[n+1,self.sink] = 1
        self.sparsity = pattern.tocsr()

    def fluxes(self, h):
        stage = self.bed + h
        slope = (stage[self.a]-stage[self.b])/self.dx
        # Smooth signed sqrt near zero; supports reversals and exact lake at rest.
        gradient = slope / (slope*slope+self.slope_epsilon**2)**.25
        depth_face = .5*(h[self.a]+h[self.b])
        q = conveyance(depth_face,self.width,self.roughness)*gradient
        outlet = conveyance(h[self.sink],self.outlet_width,self.roughness)*np.sqrt(self.outlet_slope)
        return q,float(outlet)

    def rhs(self, time, state, inflow):
        q,out = self.fluxes(state[:len(self.bed)])
        incoming = float(inflow(time))
        balance = self.incidence @ q
        balance[self.source] += incoming
        balance[self.sink] -= out
        return np.r_[balance/self.storage_area,incoming,out]

    def jacobian(self, time, state):
        """Analytic sparse Jacobian, including the two boundary-flux integrals."""
        h=state[:len(self.bed)]
        hf=.5*(h[self.a]+h[self.b])
        s=((self.bed+h)[self.a]-(self.bed+h)[self.b])/self.dx
        k=conveyance(hf,self.width,self.roughness)
        dk=np.where(hf>0,k*(5/(3*np.maximum(hf,1e-30))-4/(3*(self.width+2*np.maximum(hf,0.)))),0.)
        den=s*s+self.slope_epsilon**2
        g=s/den**.25
        dg=(.5*s*s+self.slope_epsilon**2)/den**1.25
        da=.5*dk*g+k*dg/self.dx
        db=.5*dk*g-k*dg/self.dx
        n=len(self.bed)
        rows=np.r_[self.a,self.a,self.b,self.b,self.sink,n+1]
        cols=np.r_[self.a,self.b,self.a,self.b,self.sink,self.sink]
        hs=max(h[self.sink],1e-30)
        kout=conveyance(hs,self.outlet_width,self.roughness)
        dout=kout*(5/(3*hs)-4/(3*(self.outlet_width+2*hs)))*np.sqrt(self.outlet_slope)
        values=np.r_[-da/self.storage_area[self.a],-db/self.storage_area[self.a],
                      da/self.storage_area[self.b],db/self.storage_area[self.b],
                      -dout/self.storage_area[self.sink],dout]
        return coo_matrix((values,(rows,cols)),shape=(n+2,n+2)).tocsc()

    def normal_guess(self, inflow):
        # Width allocation ONLY seeds initialization; it is never used in fluxes.
        f = self.faces.drop_duplicates('link')
        g = nx.DiGraph()
        g.add_edges_from(zip(f.original_u,f.original_v))
        source = next(n for n in g if g.in_degree(n)==0)
        flow = {source:inflow}
        guesses = np.zeros(len(self.bed))
        for u in nx.topological_sort(g):
            outgoing = f[f.original_u == u]
            for r in outgoing.itertuples():
                q = flow[u]*r.width/outgoing.width.sum()
                flow[r.original_v] = flow.get(r.original_v,0.)+q
                mask = self.faces.link.eq(r.link).to_numpy()
                slope = (self.bed[self.a[mask][0]]-self.bed[self.b[mask][-1]])/self.dx[mask].sum()
                h = normal_depth(q,r.width,max(slope,1e-7),self.roughness)
                np.add.at(guesses,self.a[mask],h*self.width[mask]*self.dx[mask]/2)
                np.add.at(guesses,self.b[mask],h*self.width[mask]*self.dx[mask]/2)
        return guesses/self.storage_area

    def integrate(self, initial, times, inflow, *, max_step=600., rtol=1e-7, atol=1e-8):
        n = len(self.bed)
        y = np.r_[initial,0.,0.]
        result = solve_ivp(lambda t,y:self.rhs(t,y,inflow), (float(times[0]),float(times[-1])), y,
            method='BDF', t_eval=times, jac=self.jacobian,
            max_step=max_step,rtol=rtol,atol=np.r_[np.full(n,atol),1e-3,1e-3])
        if not result.success:
            raise RuntimeError(result.message)
        if not np.isfinite(result.y).all() or result.y[:n].min() <= 1e-5:
            raise RuntimeError('Wet-channel prototype reached dry/invalid state; no clipping performed')
        return result

    def spinup(self, inflow, max_days=30):
        h = self.normal_guess(inflow)
        for day in range(1,max_days+1):
            sol = self.integrate(h,np.array([0.,86400.]),lambda _:inflow,max_step=3600.,rtol=1e-8)
            h = sol.y[:-2,-1]
            rhs = self.rhs(0.,np.r_[h,0.,0.],lambda _:inflow)
            residual = np.max(np.abs(rhs[:-2]*self.storage_area))/inflow
            if residual < 1e-8:
                return h,dict(days=day,maximum_node_residual_fraction=float(residual),
                              outlet_cms=self.fluxes(h)[1])
        raise RuntimeError(f'Spinup did not converge: residual {residual}')
