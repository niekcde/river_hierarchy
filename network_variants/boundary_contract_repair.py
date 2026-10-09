"""Target-boundary contracts and conservative, topology-supported repair proposals.

This module never writes production products. Saved node matches are evidence,
not authority: exterior directed paths and geometry must corroborate them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import substring, unary_union


def ids(value):
    if value is None or (not isinstance(value,(list,tuple,set)) and pd.isna(value)):return []
    if isinstance(value,(list,tuple,set)):return [int(v) for v in value]
    return [int(float(v)) for v in str(value).replace(';',',').split(',') if v.strip()]


def derive_contract(parent, selected_ids, *, declared_boundaries=None, target_ok=True):
    selected_ids=set(map(int,selected_ids))
    if not selected_ids or not selected_ids<=set(parent.id_link):raise ValueError('Selected parent links unresolved')
    inside=parent[parent.id_link.isin(selected_ids)];outside=parent[~parent.id_link.isin(selected_ids)]
    us,ds=set(inside.id_us_node),set(inside.id_ds_node);vs=us|ds
    boundary=(vs&(set(outside.id_us_node)|set(outside.id_ds_node)))|(us-ds)|(ds-us)
    entries=(us-ds)|(vs&set(outside.id_ds_node));exits=(ds-us)|(vs&set(outside.id_us_node))
    records=[]
    for p in sorted(boundary):
        records.append(dict(parent_node=int(p),entry=p in entries,exit=p in exits,
            exterior_in=outside.loc[outside.id_ds_node.eq(p),'id_link'].astype(int).tolist(),
            exterior_out=outside.loc[outside.id_us_node.eq(p),'id_link'].astype(int).tolist()))
    mismatch=declared_boundaries is not None and boundary!=set(declared_boundaries)
    return dict(selected_parent_links=sorted(selected_ids),boundaries=records,
                boundary_nodes=sorted(map(int,boundary)),entry_nodes=sorted(map(int,entries)),exit_nodes=sorted(map(int,exits)),
                target_ok=bool(target_ok),declared_boundary_mismatch=bool(mismatch),valid=bool(target_ok and not mismatch))


def orient(links,nodes):
    result=links.copy();points=nodes.set_index('id_node').geometry
    for idx,r in result.iterrows():
        g=r.geometry
        if g.geom_type!='LineString':raise ValueError('Expected LineString')
        if Point(g.coords[0]).distance(points.loc[r.id_us_node])>1e-5:g=LineString(list(g.coords)[::-1])
        if Point(g.coords[0]).distance(points.loc[r.id_us_node])>1e-5 or Point(g.coords[-1]).distance(points.loc[r.id_ds_node])>1e-5:raise ValueError('Geometry endpoints disagree with directed graph')
        result.at[idx,'geometry']=g
    return result


class PathIndex:
    """Trace directed exterior chains, allowing subdivision but no hidden junctions."""
    def __init__(self,links,mapping):
        self.links=links.set_index('id_link',drop=False);self.mapping=mapping
        self.rows={int(r.id_link):r for r in links.itertuples()}
        self.out={};self.inc={}
        for r in links.itertuples():
            self.out.setdefault(int(r.id_us_node),[]).append(int(r.id_link));self.inc.setdefault(int(r.id_ds_node),[]).append(int(r.id_link))
        self.inverse={p:c for c,p in mapping.items()}

    def paths(self,pu,pv):
        if pu not in self.inverse or pv not in self.inverse:return []
        start,end=self.inverse[pu],self.inverse[pv];found=[]
        for first in self.out.get(start,[]):
            chain=[];seen={start};lid=first
            while True:
                r=self.rows[lid];chain.append(lid);v=int(r.id_ds_node)
                if v==end:found.append(chain);break
                if v in seen or len(self.inc.get(v,[]))!=1 or len(self.out.get(v,[]))!=1:break
                seen.add(v);lid=self.out[v][0]
        return found

    def compare(self,parent_row,tolerance):
        candidates=[]
        for chain in self.paths(int(parent_row.id_us_node),int(parent_row.id_ds_node)):
            gs=[self.rows[lid].geometry for lid in chain]
            distance=parent_row.geometry.hausdorff_distance(gs[0] if len(gs)==1 else unary_union(gs))
            delta=abs(parent_row.geometry.length-sum(g.length for g in gs))
            if distance<=tolerance and delta<=2*tolerance:
                candidates.append(dict(child_links=chain,hausdorff_m=float(distance),length_error_m=float(delta)))
        return candidates


def boundary_evidence(p,contract,parent,index,tolerance):
    spec=next(x for x in contract['boundaries'] if x['parent_node']==p)
    exterior=spec['exterior_in']+spec['exterior_out'];evidence=[]
    if not exterior:return False,[], 'no_exterior_witness'
    for lid in exterior:
        pr=parent.loc[parent.id_link.eq(lid)].iloc[0]
        paths=index.compare(pr,tolerance)
        if len(paths)!=1:return False,evidence,'missing_or_ambiguous_exterior_path'
        evidence.append(dict(parent_link=lid,**paths[0]))
    used=[child for item in evidence for child in item['child_links']]
    if len(set(used))!=len(used):return False,evidence,'exterior_paths_reuse_child_reach'
    return True,evidence,''


def propose_repairs(parent,parent_nodes,child,child_nodes,mapping,contract,node_matches,*,geometry_tolerance_m,reviewed_matches=None,reviewed_cuts=()):
    """Return explicit mappings and scoped cut locations; unresolved stays explicit.

    geometry_tolerance_m is a recorded secondary geometric envelope, not a
    nearest-node matching rule. No correspondence is inferred from distance alone.
    """
    from RAPID.src.rapid_tools.anchor_repair_trial import repair
    child=orient(child,child_nodes);original=dict(mapping);mapping=dict(mapping)
    points=parent_nodes.set_index('id_node').geometry;cp=child_nodes.set_index('id_node').geometry
    records=[];accepted={};cuts={};required=contract['boundary_nodes']
    if not contract['valid']:return dict(matches={},cuts={},records=[],unresolved=required,blocked='target_or_boundary_contract_mismatch')
    claimed={}
    if {'child_id_node','matched_parent_node_id','match_within_tolerance'}<=set(node_matches):
        for r in node_matches.itertuples():
            if pd.notna(r.matched_parent_node_id) and r.match_within_tolerance==True:
                claimed.setdefault(int(r.matched_parent_node_id),set()).add(int(r.child_id_node))
    # Reviewed correspondences are explicit evidence, still checked for collisions.
    for raw,p in (reviewed_matches or {}).items():
        if raw not in mapping or p not in required:raise ValueError('Invalid reviewed correspondence')
        if p in mapping.values() and mapping[raw]!=p:raise ValueError('Reviewed correspondence collision')
        if mapping[raw]!=p:accepted[raw]=p;mapping[raw]=p
        records.append(dict(parent_node=p,raw_node=raw,decision='reviewed_correspondence',displacement_m=float(cp.loc[raw].distance(points.loc[p]))))
    pending=set(required)-set(mapping.values())
    # Recover identities iteratively; an accepted opposite boundary can witness the next.
    while True:
        updates=[]
        for p in sorted(pending-set(reviewed_cuts)):
            candidates=[]
            for raw in sorted(claimed.get(p,set())):
                if raw not in mapping or mapping[raw] in set(parent_nodes.id_node):continue
                distance=cp.loc[raw].distance(points.loc[p])
                if distance>geometry_tolerance_m:continue
                trial=dict(mapping);trial[raw]=p
                okay,evidence,reason=boundary_evidence(p,contract,parent,PathIndex(child,trial),geometry_tolerance_m)
                if okay:candidates.append((raw,distance,evidence))
            if len(candidates)==1:updates.append((p,*candidates[0]))
        if not updates:break
        proposed_raw=[z[1] for z in updates]
        for p,raw,distance,evidence in updates:
            if proposed_raw.count(raw)!=1:continue
            accepted[raw]=p;mapping[raw]=p;pending.remove(p)
            records.append(dict(parent_node=p,raw_node=raw,decision='exterior_supported_saved_correspondence',displacement_m=float(distance),evidence=evidence))
        if all(proposed_raw.count(z[1])!=1 for z in updates):break
    # Missing cuts: only candidate links that also reconstruct the required exterior
    # paths can qualify. Multiple qualifying links are never resolved by nearest distance.
    for p in sorted(pending):
        if p not in reviewed_cuts and any(raw in cp.index and cp.loc[raw].distance(points.loc[p])<=geometry_tolerance_m for raw in claimed.get(p,set())):
            records.append(dict(parent_node=p,decision='unresolved_saved_correspondence'));continue
        hits=[]
        for i in child.sindex.query(points.loc[p].buffer(geometry_tolerance_m),predicate='intersects'):
            rr=child.iloc[i];g=rr.geometry;d=g.project(points.loc[p])
            if not 1e-5<d<g.length-1e-5:continue
            projection=g.interpolate(d)
            if p in reviewed_cuts and projection.distance(points.loc[p])>1e-5:continue
            projected_parent=parent_nodes.copy();projected_parent.loc[projected_parent.id_node.eq(p),'geometry']=projection
            try:
                out,ns,actions=repair(child,child_nodes,projected_parent,mapping,required,cuts=[p],cut_link_ids={p:int(rr.id_link)})
                mp=dict(zip(ns.id_node,ns.parent_identity))
                okay,evidence,reason=boundary_evidence(p,contract,parent,PathIndex(out,mp),geometry_tolerance_m)
                if okay or p in reviewed_cuts:hits.append((int(rr.id_link),projection,float(projection.distance(points.loc[p])),evidence))
            except ValueError:continue
        if len(hits)==1:
            lid,point,distance,evidence=hits[0];cuts[p]=dict(link_id=lid,x=point.x,y=point.y)
            records.append(dict(parent_node=p,source_link=lid,decision='reviewed_cut' if p in reviewed_cuts else 'exterior_supported_cut',displacement_m=distance,evidence=evidence))
        else:records.append(dict(parent_node=p,decision='unresolved_cut_location',candidate_count=len(hits)))
    unresolved=sorted(set(required)-set(mapping.values())-set(cuts))
    return dict(matches=accepted,cuts=cuts,records=records,unresolved=unresolved,blocked='')


def replacement_connections(parent,after,mapping,contract,exterior_edges):
    selected=parent[parent.id_link.isin(contract['selected_parent_links'])]
    pg=nx.DiGraph();pg.add_edges_from(zip(selected.id_us_node,selected.id_ds_node))
    replacement=after[~after.id_link.isin(exterior_edges)]
    cg=nx.DiGraph();cg.add_edges_from((mapping[int(r.id_us_node)],mapping[int(r.id_ds_node)]) for r in replacement.itertuples())
    records=[]
    for entry in contract['entry_nodes']:
        expected=nx.descendants(pg,entry) if entry in pg else set()
        actual=nx.descendants(cg,entry) if entry in cg else set()
        for exit in contract['exit_nodes']:
            if entry==exit:continue
            records.append(dict(entry=entry,exit=exit,expected=exit in expected,actual=exit in actual))
    return records


def validate_result(parent,before,after,mapping,contract,*,geometry_tolerance_m):
    index=PathIndex(after,mapping);exterior=[]
    for pr in parent[~parent.id_link.isin(contract['selected_parent_links'])].itertuples():
        matches=index.compare(pr,geometry_tolerance_m)
        exterior.append(dict(parent_link=int(pr.id_link),path_unique=len(matches)==1,
            geometry_exact=len(matches)==1 and matches[0]['hausdorff_m']<=1e-5 and matches[0]['length_error_m']<=1e-5,
            **(matches[0] if len(matches)==1 else {'child_links':[]})))
    exterior_edges={lid for r in exterior if r['path_unique'] for lid in r['child_links']}
    assignments={}
    for r in exterior:
        if r['path_unique']:
            for lid in r['child_links']:assignments.setdefault(lid,[]).append(r['parent_link'])
    exterior_reuse={int(k):v for k,v in assignments.items() if len(set(v))>1}
    connections=replacement_connections(parent,after,mapping,contract,exterior_edges)
    boundary=[]
    for spec in contract['boundaries']:
        p=spec['parent_node'];raw=index.inverse.get(p)
        ok,evidence,reason=boundary_evidence(p,contract,parent,index,geometry_tolerance_m) if raw is not None else (False,[],'missing_identity')
        if raw is not None and not spec['exterior_in'] and not spec['exterior_out']:ok=True;reason='network_terminal'
        incoming=[lid for lid in index.inc.get(raw,[]) if lid not in exterior_edges]
        outgoing=[lid for lid in index.out.get(raw,[]) if lid not in exterior_edges]
        incidence=raw is not None and (not spec['entry'] or bool(outgoing)) and (not spec['exit'] or bool(incoming))
        boundary.append(dict(parent_node=p,present=raw is not None,exterior_supported=ok,replacement_incidence=incidence,reason=reason))
    def signature(f):
        g=nx.MultiDiGraph();g.add_edges_from(zip(f.id_us_node,f.id_ds_node))
        return (nx.is_directed_acyclic_graph(g),nx.number_weakly_connected_components(g),g.number_of_edges()-g.number_of_nodes(),sorted(v for v in g if g.in_degree(v)==0),sorted(v for v in g if g.out_degree(v)==0))
    topology_preserved=signature(before)==signature(after)
    geometry_preserved=True
    old=before.set_index('id_link')
    sizes=after.groupby('source_link').size()
    singles=after[after.source_link.isin(sizes[sizes.eq(1)].index)]
    if len(singles):
        from shapely import hausdorff_distance
        original=old.loc[singles.source_link].geometry
        geometry_preserved=bool((np.abs(singles.length.to_numpy()-original.length.to_numpy())<=1e-5).all()
                                and (hausdorff_distance(singles.geometry.to_numpy(),original.to_numpy())<=1e-5).all())
    for source,fragment in after[after.source_link.isin(sizes[sizes.gt(1)].index)].groupby('source_link'):
        geom=old.loc[source].geometry
        if abs(fragment.length.sum()-geom.length)>1e-5 or geom.hausdorff_distance(unary_union(fragment.geometry))>1e-5:geometry_preserved=False
    return dict(boundaries=boundary,exterior=exterior,topology_preserved=topology_preserved,geometry_preserved=geometry_preserved,
        exterior_path_reuse=exterior_reuse,
        replacement_connections=connections,
        replacement_connections_preserved=all(z['expected']==z['actual'] for z in connections),
        anchor_pass=not exterior_reuse and all(z['present'] and z['exterior_supported'] and z['replacement_incidence'] for z in boundary),
        exterior_exact=not exterior_reuse and all(z['geometry_exact'] for z in exterior))


def paired_exterior_cuts(parent,parent_nodes,child,child_nodes,mapping,contract,*,geometry_tolerance_m):
    """Propose two cuts together when a preserved exterior reach witnesses both.

    A single-endpoint test cannot establish its exterior path until the opposite
    missing endpoint also exists. Accept only a unique same-link pair, with both
    boundaries' complete exterior-path checks passing after provisional cuts.
    """
    from RAPID.src.rapid_tools.anchor_repair_trial import repair
    points=parent_nodes.set_index('id_node').geometry
    missing=set(contract['boundary_nodes'])-set(mapping.values());proposals=[]
    exterior=parent[~parent.id_link.isin(contract['selected_parent_links'])]
    for pr in exterior.itertuples():
        u,v=int(pr.id_us_node),int(pr.id_ds_node)
        if not {u,v}<=missing:continue
        candidates=[]
        for cr in child.itertuples():
            line=cr.geometry;du,dv=line.project(points.loc[u]),line.project(points.loc[v])
            if not 1e-5<du<dv<line.length-1e-5:continue
            pu,pv=line.interpolate(du),line.interpolate(dv)
            if max(pu.distance(points.loc[u]),pv.distance(points.loc[v]))>geometry_tolerance_m:continue
            pp=parent_nodes.copy();pp.loc[pp.id_node.eq(u),'geometry']=pu;pp.loc[pp.id_node.eq(v),'geometry']=pv
            trial,nodes,_=repair(child,child_nodes,pp,mapping,contract['boundary_nodes'],cuts=[u,v],cut_link_ids={u:int(cr.id_link),v:int(cr.id_link)})
            mp=dict(zip(nodes.id_node,nodes.parent_identity));idx=PathIndex(trial,mp)
            checks=[boundary_evidence(n,contract,parent,idx,geometry_tolerance_m) for n in [u,v]]
            if all(check[0] for check in checks):
                candidates.append(dict(parent_exterior_link=int(pr.id_link),cuts={u:dict(link_id=int(cr.id_link),x=pu.x,y=pu.y),v:dict(link_id=int(cr.id_link),x=pv.x,y=pv.y)},evidence=[z[1] for z in checks]))
        if len(candidates)==1:proposals.append(candidates[0])
    return proposals
