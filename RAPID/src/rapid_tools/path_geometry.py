"""Path descriptors on a DAG without enumerating its source-to-sink paths.

Uniform-path expectations use suffix path counts. Harmonic-width sums use
1/R = integral exp(-t R) dt with checked positive numerical quadrature.
Definitions match the existing enumerated-path descriptors; shared edges count
once on every path containing them, not as independent physical channels.
"""
import math
import networkx as nx
import numpy as np
from numpy.polynomial.legendre import leggauss


def path_geometry(frame, ids):
    f=frame[frame.id_link.isin(ids)]
    if len(f)!=len(set(ids)):raise ValueError('Missing or duplicate domain edges')
    g=nx.MultiDiGraph()
    for r in f.itertuples():
        if not (r.link_length_m>0 and r.width>0):raise ValueError('Nonpositive geometry')
        g.add_edge(r.id_us_node,r.id_ds_node,key=int(r.id_link),length=float(r.link_length_m),width=float(r.width))
    sources=[n for n in g if g.in_degree(n)==0];sinks=[n for n in g if g.out_degree(n)==0]
    if len(sources)!=1 or len(sinks)!=1 or not nx.is_directed_acyclic_graph(g):raise ValueError('Domain needs one source/sink and no cycle')
    source,sink=sources[0],sinks[0];order=list(reversed(list(nx.topological_sort(g))))
    counts={sink:1};low_l={sink:0.};low_r={sink:0.};high_r={sink:0.};mean={sink:0.};var={sink:0.};out={}
    for u in order:
        if u==sink:continue
        edges=[(v,d['length'],d['length']/d['width']) for _,v,k,d in g.out_edges(u,keys=True,data=True)]
        counts[u]=sum(counts[v] for v,l,r in edges)
        out[u]=[(v,l,r,counts[v]/counts[u]) for v,l,r in edges]
        low_l[u]=min(l+low_l[v] for v,l,r in edges);low_r[u]=min(r+low_r[v] for v,l,r in edges);high_r[u]=max(r+high_r[v] for v,l,r in edges)
        mean[u]=sum(p*(l+mean[v]) for v,l,r,p in out[u])
        var[u]=sum(p*(var[v]+(l+mean[v]-mean[u])**2) for v,l,r,p in out[u])
    heads={**dict(zip(f.id_us_node,f.head_us)),**dict(zip(f.id_ds_node,f.head_ds))}
    if counts[source]==1:
        # Preserve exact single-path invariants instead of feeding quadrature
        # roundoff in dominance/association to a threshold-based predictor.
        return dict(equivalent_length=mean[source],elongation=low_r[source],
            dominant_width_fraction=1.,path_length_cv=0.,association=0.,
            slope=(heads[source]-heads[sink])/low_l[source],paths=1.,
            quadrature_relative_error=0.),source,sink
    # Maximum L/R path through fractional optimization; no path enumeration.
    ratio=float(f.width.min())
    for iteration in range(100):
        best={sink:(0.,0.)}
        for u in order:
            if u!=sink:best[u]=max(((l+best[v][0],r+best[v][1]) for v,l,r,p in out[u]),key=lambda x:x[0]-ratio*x[1])
        new=best[source][0]/best[source][1]
        if abs(new-ratio)<=1e-12*max(1.,abs(new)):break
        ratio=new
    else:raise ValueError('Maximum path width did not converge')
    max_width=new;ls=low_l[source];rs=low_r[source]
    a=math.log(1e-12*rs/high_r[source]);b=math.log(40.);bounds=np.linspace(a,b,math.ceil(b-a)+1)
    def integral(degree):
        nodes,weights=leggauss(degree);z=((bounds[:-1,None]+bounds[1:,None])/2+(bounds[1:,None]-bounds[:-1,None])/2*nodes).ravel()
        t=np.exp(z);w=((bounds[1:,None]-bounds[:-1,None])/2*weights).ravel()*t
        moments={sink:(np.ones(len(t)),np.zeros(len(t)),np.zeros(len(t)))}
        for u in order:
            if u==sink:continue
            z0=np.zeros(len(t));z1=z0.copy();z2=z0.copy()
            for v,l,r,p in out[u]:
                m0,m1,m2=moments[v];x=l/ls;factor=p*np.exp(-t*r/rs)
                z0+=factor*m0;z1+=factor*(m1+x*m0);z2+=factor*(m2+2*x*m1+x*x*m0)
            moments[u]=(z0,z1,z2)
        _,m1,m2=moments[source]
        return np.array([np.dot(w,m1)*ls/rs,np.dot(w,m2)*ls*ls/rs])
    first=integral(8);second=integral(16)
    relative=float(np.max(abs(first-second)/np.maximum(abs(second),1e-300)))
    if relative>1e-8:
        third=integral(32);relative=float(np.max(abs(second-third)/np.maximum(abs(third),1e-300)));second=third
    if relative>1e-8:raise ValueError(f'Path quadrature failed convergence: {relative}')
    ew,elw=second;leq=elw/ew;weq=float(counts[source])*ew
    association=leq/mean[source]-1
    if abs(association)<1e-12:association=0.
    result=dict(equivalent_length=leq,elongation=leq/weq,dominant_width_fraction=max_width/weq,path_length_cv=math.sqrt(var[source])/mean[source],association=association,slope=(heads[source]-heads[sink])/ls,paths=float(counts[source]),quadrature_relative_error=relative)
    if not np.isfinite(list(result.values())).all():raise ValueError('Nonfinite path descriptors')
    return result,source,sink
