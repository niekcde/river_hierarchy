"""Reconstruct changed domains with shared heads and fixed outer anchors.

Geometry and reference-profile anchors determine the solve. No celerity,
roughness, routing clock or stability target enters this module.
"""
from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.linalg import spsolve, factorized


def _constrained_least_squares(C, b, lower, initial):
    """Convex relative-slope least squares with a primal/dual residual audit."""
    C=sparse.csc_matrix(C)
    P=2*(C.T@C);q=2*np.asarray(C.T@(b-1)).ravel()
    scale=max(float(abs(P).max()),float(np.max(abs(q))),1.)
    P=P/scale;q=q/scale
    x=spsolve(P,-q)
    residual=np.asarray(C@x+b)
    if np.isfinite(x).all() and residual.min()>=lower-1e-9:
        return x,dict(method='sparse_unconstrained',iterations=0,primal_residual=0.,stationarity=float(np.max(abs(P@x+q))))
    # Scale constraints independently to dimensionless head differences.
    row_scale=np.asarray(abs(C).max(axis=1).toarray()).ravel()
    A=sparse.diags(1/row_scale)@C;lo=(lower-b)/row_scale
    feasible=linprog(np.zeros(C.shape[1]),A_ub=-A,b_ub=-lo,bounds=[(None,None)]*C.shape[1],method='highs')
    if not feasible.success:raise ValueError('Infeasible boundary heads and relative slope floor')
    x=feasible.x;rho=1.;At=A.T.tocsc();sigma=1e-10
    z=np.maximum(A@x,lo);y=np.zeros(len(z))
    def factor():return factorized((P+rho*(At@A)+sigma*sparse.eye(len(x))).tocsc())
    solve=factor()
    for it in range(60000):
        x=solve(-q+At@(rho*z-y)+sigma*x)
        ax=np.asarray(A@x);old=z.copy();z=np.maximum(ax+y/rho,lo);y+=rho*(ax-z)
        primal=float(np.max(abs(ax-z)));stationarity=float(np.max(abs(P@x+q+At@y)))
        if primal<1e-10 and stationarity<1e-9:
            if (C@x+b).min()<lower-1e-6:raise RuntimeError('Relative slope check failed after convergence')
            return x,dict(method='sparse_admm',iterations=it+1,primal_residual=primal,stationarity=stationarity)
        if (it+1)%100==0:
            dual=float(np.max(abs(rho*At@(z-old))))
            new=rho*2 if primal>10*dual else rho/2 if dual>10*primal else rho
            new=float(np.clip(new,1e-8,1e6))
            if new!=rho:rho=new;solve=factor()
    raise RuntimeError(f'Head allocation did not converge: primal={primal}, stationarity={stationarity}')


def reconstruct_shared_heads(links, base_links, base_nodes, mapped_heads, *, floor_fraction=.01, fixed_node_ids=()):
    """Keep unchanged domains; jointly re-solve interiors of changed domains.

    `links` has u,v,link_length_m; base_links has id_us_node,id_ds_node,
    actual_length_m,ownership_kind,ownership_domain_id. Matched child nodes use
    base IDs; genuinely new vertices must have fresh IDs. Reference anchors
    and positions come from base_nodes. Width is intentionally absent.
    `fixed_node_ids` optionally protects additional reference heads, e.g. all
    exterior nodes in an isolated boundary-contract trial. Defaults are unchanged.
    """
    f=links.copy();heads=mapped_heads.copy();bn=base_nodes.set_index('id_node');base=base_links
    explicit_fixed=set(map(int,fixed_node_ids))
    if not explicit_fixed<=set(heads.index):raise ValueError('Explicit fixed node lacks a reference head')
    unit_mask=base.ownership_kind.isin(['simple_unit','compound_bubble'])
    units=[];protected=set()
    for _,g in base[unit_mask].groupby(['ownership_kind','ownership_domain_id']):
        G=nx.DiGraph();G.add_edges_from(zip(g.id_us_node,g.id_ds_node))
        boundary={v for v in G if G.in_degree(v)==0 or G.out_degree(v)==0}
        protected|=boundary;units.append((set(G),boundary))
    direct=base[~unit_mask];protected|=set(direct.id_us_node)|set(direct.id_ds_node)
    old={}
    for r in base.itertuples():old.setdefault((r.id_us_node,r.id_ds_node),[]).append(r.actual_length_m)
    changed=[]
    for r in f.itertuples():
        if not any(np.isclose(r.link_length_m,L,rtol=1e-8,atol=1e-5) for L in old.get((r.u,r.v),[])):
            changed.append((r.u,r.v))
    nodes=set(f.u)|set(f.v);unknown=nodes-set(heads.index)
    affected={v for edge in changed for v in edge}
    free=set(unknown)
    # A link inside an owned domain invalidates that domain's interior heads.
    for vs,boundary in units:
        if affected&(vs-boundary):free|=(vs-protected)&nodes
        elif any(u in vs and v in vs for u,v in changed):free|=(vs-protected)&nodes
    free|=(affected-protected)&nodes
    free-=explicit_fixed
    graph=nx.DiGraph();graph.add_edges_from(zip(f.u,f.v))
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Child graph is not acyclic')
    # All original exterior/profile anchors remain hard constraints.
    released=[]
    audits=[]
    for component in nx.connected_components(graph.subgraph(free).to_undirected()):
        patch=f[f.u.isin(component)|f.v.isin(component)];boundary=(set(patch.u)|set(patch.v))-component
        if len(boundary)<2:raise ValueError('Changed patch lacks exterior anchors')
        boundary_heads=heads.loc[list(boundary)];span=float(boundary_heads.max()-boundary_heads.min())
        positions=bn.loc[list(boundary),'reference_section_dist_out_m'];extent=float(positions.max()-positions.min())
        if span<=0 or extent<=0:raise ValueError('Nonpositive boundary span or reference extent')
        reference=span/extent;center=float(boundary_heads.min());ids=sorted(component);ix={v:i for i,v in enumerate(ids)}
        C=sparse.lil_matrix((len(patch),len(ids)));b=np.zeros(len(patch));scales=reference*patch.link_length_m.to_numpy()/span
        for j,r in enumerate(patch.itertuples()):
            for v,sign in [(r.u,1),(r.v,-1)]:
                if v in ix:C[j,ix[v]]+=sign/scales[j]
                else:b[j]+=sign*(heads.loc[v]-center)/span/scales[j]
        initial=np.array([(heads.get(v,center+.5*span)-center)/span for v in ids])
        values,qa=_constrained_least_squares(C.tocsc(),b,floor_fraction,initial)
        for v,value in zip(ids,values):heads.loc[v]=center+span*value
        audits.append(dict(n_links=len(patch),n_free_nodes=len(ids),boundary_nodes=sorted(boundary),reference_slope=reference,head_span_m=span,**qa))
    f['head_us']=f.u.map(heads);f['head_ds']=f.v.map(heads);f['slope']=(f.head_us-f.head_ds)/f.link_length_m
    if not np.isfinite(f.slope).all() or (f.slope<=0).any():raise ValueError('Nonpositive child drop after domain reconstruction: '+str(f.loc[f.slope.le(0),['u','v','link_length_m','head_us','head_ds']].to_dict('records')))
    fixed=nodes-free
    assert all(heads.loc[v]==mapped_heads.loc[v] for v in fixed)
    return f,dict(changed_links=len(changed),free_nodes=len(free),released_interior_anchors=released,patches=audits)


def match_child_nodes(base_nodes, child_nodes, base_links, child_links, *, tolerance_m=1e-5, validated_correspondences=None):
    """Injective coordinate matching, resolving coincident child ports by topology.

    A raster collapse can create two distinct graph nodes at the same point.
    Prefer an unchanged direct-profile continuation, then the most unchanged
    incident links, then greatest unchanged incident length. That node inherits
    the base head. The other is a genuinely new hydraulic node, not a self-loop.
    Ambiguous ties fail explicitly; numeric node-ID equality is not evidence.
    `validated_correspondences` optionally supplies raw-child -> original-parent
    IDs already justified by boundary-contract/topology checks. These survive
    coordinate rematching. Contradictory or colliding identities fail explicitly;
    do not pass an unvalidated nearest-node matching table here.
    """
    from scipy.spatial import cKDTree
    bx=np.c_[base_nodes.geometry.x,base_nodes.geometry.y]
    if base_nodes.geometry.duplicated().any():raise ValueError('Base coordinates are not unique')
    distance,ix=cKDTree(bx).query(np.c_[child_nodes.geometry.x,child_nodes.geometry.y])
    ids=child_nodes.id_node.astype(int).to_numpy();parent=base_nodes.id_node.astype(int).to_numpy()[ix]
    candidates={int(v):int(p) for v,p,d in zip(ids,parent,distance) if d<=tolerance_m}
    explicit={int(c):int(p) for c,p in (validated_correspondences or {}).items()}
    if not set(explicit)<=set(ids) or not set(explicit.values())<=set(base_nodes.id_node):
        raise ValueError('Validated correspondence refers to an absent node')
    if len(set(explicit.values()))!=len(explicit):
        raise ValueError('Validated correspondences are not injective')
    for c,p in explicit.items():
        if c in candidates and candidates[c]!=p:
            raise ValueError('Validated correspondence contradicts an exact coordinate identity')
        if any(other!=c and old==p for other,old in candidates.items()):
            raise ValueError('Validated correspondence collides with an exact coordinate identity')
    candidates.update(explicit)
    old={}
    for r in base_links.itertuples():
        direct=getattr(r,'ownership_kind','simple_unit') not in ['simple_unit','compound_bubble']
        old.setdefault((int(r.id_us_node),int(r.id_ds_node)),[]).append((r.actual_length_m,direct))
    score={v:(0,0,0.) for v in candidates}
    for r in child_links.itertuples():
        u,v=int(r.id_us_node),int(r.id_ds_node)
        if u not in candidates or v not in candidates:continue
        matches=[(L,direct) for L,direct in old.get((candidates[u],candidates[v]),[]) if np.isclose(r.link_length_m,L,rtol=1e-8,atol=tolerance_m)]
        if matches:
            direct=int(any(z[1] for z in matches))
            for node in [u,v]:
                a,b,length=score[node];score[node]=(a+direct,b+1,length+r.link_length_m)
    by_parent={}
    for v,p in candidates.items():by_parent.setdefault(p,[]).append(v)
    collisions=[]
    for p,vs in by_parent.items():
        if len(vs)==1:continue
        best=max(score[v] for v in vs);winners=[v for v in vs if score[v]==best]
        if len(winners)!=1:raise ValueError(f'Ambiguous coincident child nodes at base node {p}: {vs}')
        for v in vs:
            if v!=winners[0]:del candidates[v]
        collisions.append(dict(base_node=p,inheriting_child=winners[0],new_child_nodes=[v for v in vs if v!=winners[0]]))
    mapping=dict(candidates);unknown=set();next_id=int(base_nodes.id_node.max())+1
    for v in ids:
        if v not in mapping:mapping[int(v)]=next_id;unknown.add(next_id);next_id+=1
    assert len(set(mapping.values()))==len(mapping)
    return mapping,unknown,dict(coincident_ports=collisions,max_nearest_distance_m=float(distance.max()),validated_correspondences=explicit)
