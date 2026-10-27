"""Seven geometric descriptors of selected regions, with connectors excluded.

Keep the connected graph for allocation; assign zero length reward to excluded
edges. Width moments use only selected edges and their physical lengths. Split
statistics use only splits wholly inside selected regions. No hydraulic inputs.
"""
import numpy as np
import pandas as pd
import networkx as nx
from geometry_audit import pilot


def describe_edited(frame,selected_ids):
    f=frame.reset_index(drop=True);selected=set(map(int,selected_ids));assert selected and selected<=set(f.id_link)
    g=nx.MultiDiGraph();out={}
    for i,r in enumerate(f.itertuples()):
        u,v=int(r.id_us_node),int(r.id_ds_node);g.add_edge(u,v,key=int(r.id_link));out.setdefault(u,[]).append((i,v))
    roots=[u for u in g if g.in_degree(u)==0];ends=[u for u in g if g.out_degree(u)==0]
    if len(roots)!=1 or len(ends)!=1 or not nx.is_weakly_connected(g) or not nx.is_directed_acyclic_graph(g):raise ValueError('Connected single-source/sink DAG required')
    chosen=f.id_link.isin(selected).to_numpy();L=f.link_length_m.to_numpy(float);W=f.width.to_numpy(float);reward=np.where(chosen,L,0.)
    assert (L>0).all() and (W>0).all()
    order=list(nx.topological_sort(g));visit={roots[0]:1.};probs={};split_weight=[];dominances=[];H=0.
    for u in order:
        ee=out.get(u,[])
        if not ee:continue
        inds=[i for i,v in ee];pr=W[inds]/W[inds].sum();probs[u]=pr
        if len(ee)>1:
            mask=chosen[inds]
            if mask.any() and not mask.all():raise ValueError('Mask cuts through a split')
            if mask.all():H+=visit[u]*float(-(pr*np.log(pr)).sum());split_weight.append(visit[u]);dominances.append(float(pr.max()))
        for p,(i,v) in zip(pr,ee):visit[v]=visit.get(v,0.)+visit[u]*p
    assert np.isclose(visit[ends[0]],1.,atol=1e-12)
    mean={ends[0]:0.};variance={ends[0]:0.}
    for u in reversed(order[:-1]):
        ee=out[u];branches=np.array([reward[i]+mean[v] for i,v in ee]);pr=probs[u];mean[u]=float(np.dot(pr,branches))
        variance[u]=float(sum(p*(variance[v]+(z-mean[u])**2) for p,z,(i,v) in zip(pr,branches,ee)))
    weights=L[chosen];widths=W[chosen];wm=float(np.average(widths,weights=weights));ws=float(np.sqrt(np.average((widths-wm)**2,weights=weights)))
    return dict(route_length_mean=mean[roots[0]],route_length_sd=float(np.sqrt(variance[roots[0]])),width_mean=wm,width_sd=ws,width_cv=ws/wm,route_entropy=H,dominance_mean=float(np.average(dominances,weights=split_weight)) if split_weight else np.nan,total_length=float(weights.sum()))


def pair(parent,child,parent_ids,child_ids):
    p,c=describe_edited(parent,parent_ids),describe_edited(child,child_ids);result=pilot.pair_features(p,c,'B_')
    result['domain_size']=p['route_length_mean']/1000
    for k in ['route_length_mean','route_length_sd','width_mean','width_cv','route_entropy']:result['log1p__B_ratio_'+k]=np.log1p(result['B_ratio_'+k])
    return result
