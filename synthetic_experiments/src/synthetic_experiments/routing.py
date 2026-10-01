"""Synthetic input adapter to shared RAPID segmentation, K, and routing engine."""
from dataclasses import dataclass
import math
import time

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import factorized
from shapely.geometry import LineString, Point

from rapid_tools.k_values import KValueConfig, compute_k_values, compute_dt_bounds
from rapid_tools.prep import _compute_subreach_count, split_links_into_subreaches
from rapid_tools.engine import ccc_mat, rte_mat, mus_rte


@dataclass(frozen=True)
class Settings:
    target_length_m: float = 200
    effective_length_floor_m: float = 200
    kb: float = 20
    x: float = .1
    n_manning: float = .35
    min_celerity_mps: float = .05
    max_celerity_mps: float = 2.524
    chezy_cz: float = 22
    baseflow_cms: float = 100
    peak_cms: float = 1000
    rise_hours: float = 12
    fall_hours: float = 24
    window_travel_multiplier: float = 3
    minimum_buffer_hours: float = 48

    def __post_init__(self):
        for name,value in vars(self).items():
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0<self.x<.5 or self.peak_cms<=self.baseflow_cms:
            raise ValueError('Require 0<X<0.5 and peak>baseflow')
        if self.min_celerity_mps>self.max_celerity_mps:
            raise ValueError('Celerity bounds reversed')


def k_config(s):
    return KValueConfig(width_field='width',x_value=s.x,kb_value=s.kb,n_manning=s.n_manning,
                        min_effective_length_m=s.effective_length_floor_m,
                        use_celerity_capping=True,min_celerity_mps=s.min_celerity_mps,
                        max_celerity_mps=s.max_celerity_mps)


def compact_k(network,s):
    rows=[]
    for u,v,length,width,slope in network['edges']:
        count=_compute_subreach_count(length,s.target_length_m)
        rows.append(dict(u=u,v=v,link_length_m=length/count,width=width,slope_used=slope,count=count))
    return compute_k_values(pd.DataFrame(rows),config=k_config(s))


def schedule(networks,s,cache=None):
    cache = {} if cache is None else cache
    from .networks import digest
    low,high,travel=0.,float('inf'),0.
    for network in networks:
        key=digest(network)
        if key not in cache:
            cache[key]=compact_k(network,s)
        frame=cache[key]
        a,b=compute_dt_bounds(frame.rapid_k,x_value=s.x)
        low,high=max(low,a),min(high,b)
        graph=nx.MultiDiGraph()
        for r in frame.itertuples():
            graph.add_edge(r.u,r.v,k=r.rapid_k*r.count)
        cumulative={}
        for node in nx.topological_sort(graph):
            cumulative[node]=max((cumulative[u]+d['k'] for u,_,d in graph.in_edges(node,data=True)),default=0.)
        travel=max(travel,max(cumulative.values()))
    dt=math.ceil(low+1e-8)
    if dt>math.floor(high):
        raise ValueError(f'No common stable integer timestep: [{low}, {high}]')
    # Round to hours to increase reuse while remaining at least the required duration.
    buffer_hours=math.ceil(max(s.minimum_buffer_hours,s.window_travel_multiplier*travel/3600))
    return dict(dt_seconds=dt,dt_min_seconds=low,dt_max_seconds=high,
                buffer_hours=buffer_hours,max_path_k_seconds=travel)


def prepare(network,s):
    """Schematic line geometry is auxiliary; physical length is explicit."""
    nodes=gpd.GeoDataFrame([dict(id_node=i,is_inlet=i==0,is_outlet=i==len(network['nodes'])-1,
                                geometry=Point(x,0)) for i,(x,h) in enumerate(network['nodes'])])
    links=gpd.GeoDataFrame([dict(id_link=i,id_us_node=u,id_ds_node=v,link_length_m=l,width=w,
                                slope_used=slope,is_inlet=u==0,is_outlet=v==len(nodes)-1,
                                geometry=LineString([(network['nodes'][u][0],0),(network['nodes'][v][0],0)]))
                           for i,(u,v,l,w,slope) in enumerate(network['edges'],1)])
    segments,virtual_nodes=split_links_into_subreaches(links,nodes,target_length_m=s.target_length_m)
    segments=compute_k_values(segments,config=k_config(s)).reset_index(drop=True)
    # Every subreach retains its parent branch width; no mouth-sample dependence.
    segments['rapid_branch_width_m']=segments['width']
    return segments


def buffer_seconds(clock):
    # Keep pulse onset on a timestep boundary so longer spin-up does not shift
    # the sampling phase and masquerade as a hydraulic timing change.
    dt=clock['dt_seconds']
    return math.ceil(clock['buffer_hours']*3600/dt)*dt


def forcing(clock,s):
    dt=clock['dt_seconds']; buffer=buffer_seconds(clock)
    rise=s.rise_hours*3600; fall=s.fall_hours*3600
    duration=2*buffer+rise+fall
    t=np.arange(math.ceil(duration/dt),dtype=float)*dt
    # Exact interval-average triangular pulse; volume is independent of dt.
    def primitive(x):
        a=np.clip(x,0,rise)
        b=np.clip(x-rise,0,fall)
        return a*a/(2*rise)+b-b*b/(2*fall)
    fraction=(primitive(t+dt-buffer)-primitive(t-buffer))/dt
    q=s.baseflow_cms+(s.peak_cms-s.baseflow_cms)*fraction
    return t,q


def route(network,s,clock):
    start=time.perf_counter()
    frame=prepare(network,s)
    dt=clock['dt_seconds']
    lower,upper=compute_dt_bounds(frame.rapid_k,x_value=s.x)
    if not lower-1e-8<=dt<=upper+1e-8:
        raise ValueError('Prepared segments violate selected timestep')
    edges=list(frame.itertuples())
    outgoing={}
    for i,r in enumerate(edges): outgoing.setdefault(r.id_us_node,[]).append(i)
    rr,cc,weights=[],[],[]
    for i,r in enumerate(edges):
        downstream=outgoing.get(r.id_ds_node,[])
        total=sum(edges[j].rapid_branch_width_m for j in downstream)
        for j in downstream:
            rr.append(j); cc.append(i); weights.append(edges[j].rapid_branch_width_m/total)
    net=csc_matrix((weights,(rr,cc)),shape=(len(frame),len(frame)))
    linear,external,previous=rte_mat(net,*ccc_mat(frame.rapid_k.to_numpy(),frame.rapid_x.to_numpy(),dt))
    solver=factorized(linear)
    inlet=np.flatnonzero(frame.is_inlet.to_numpy(dtype=bool))
    outlet=np.flatnonzero(frame.is_outlet.to_numpy(dtype=bool))
    if len(inlet)!=1 or len(outlet)!=1: raise ValueError('Expected one inlet/outlet segment')
    t,q=forcing(clock,s)
    state=np.zeros(len(frame)); boundary=np.zeros(len(frame)); out=np.zeros(len(t))
    for i,value in enumerate(q):
        boundary[inlet[0]]=value
        avg,state=mus_rte(linear,external,previous,1,state,boundary,solver)
        out[i]=avg[outlet[0]]
    event_start=buffer_seconds(clock)
    event=t>=event_start
    event_ids=np.flatnonzero(event)
    peak_index=int(event_ids[np.argmax(out[event])]); inlet_peak=int(np.argmax(q))
    volume_in=float(np.sum(q[event]-s.baseflow_cms)*dt)
    volume_out=float(np.sum(out[event]-s.baseflow_cms)*dt)
    pre_index=max(0,int(np.searchsorted(t,event_start))-1)
    error=(volume_out-volume_in)/volume_in
    metrics=dict(peak_lag_seconds=float(t[peak_index]-t[inlet_peak]),peak_cms=float(out[peak_index]),
                 event_volume_in_m3=volume_in,event_volume_out_m3=volume_out,
                 volume_error_fraction=error,pre_event_relative_error=float(abs(out[pre_index]-s.baseflow_cms)/s.baseflow_cms),
                 final_relative_error=float(abs(out[-1]-s.baseflow_cms)/s.baseflow_cms),
                 min_outlet_cms=float(out.min()),n_segments=len(frame),
                 n_celerity_capped=int(frame.rapid_celerity_capped.sum()),
                 n_length_floored=int(frame.rapid_length_floor_applied.sum()),
                 wall_seconds=time.perf_counter()-start,dt_seconds=dt,
                 buffer_hours=event_start/3600)
    if not np.isfinite(out).all() or out.min() < -1e-8:
        raise ValueError('Nonfinite or negative outlet flow')
    if max(metrics['pre_event_relative_error'],metrics['final_relative_error'])>1e-4 or abs(error)>1e-4:
        raise ValueError(f'Forcing window/conservation check failed: {metrics}')
    hydro=pd.DataFrame(dict(time_seconds=t,inlet_cms=q,outlet_cms=out))
    return hydro,metrics,frame
