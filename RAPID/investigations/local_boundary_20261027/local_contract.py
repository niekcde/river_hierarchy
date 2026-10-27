"""Isolated strict-local prototype. No production writers or implicit scope expansion."""
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linprog, minimize, LinearConstraint
from scipy import sparse
from rapid_tools.cunge import parameters


class ContractError(ValueError):
    pass


def exterior_map(parent, child, parent_geometry, child_geometry, selected):
    """Require unique directed, exact-geometry exterior reaches, independent of IDs/width."""
    if not set(selected) <= set(parent.id_link):
        raise ContractError('Unresolved selected parent links')
    pg=parent_geometry.set_index('id_link').geometry
    cg=child_geometry.set_index('id_link').geometry
    mapping={};missing=[]
    for p in parent[~parent.id_link.isin(selected)].itertuples():
        candidates=child[child.u.eq(p.u)&child.v.eq(p.v)]
        hits=[int(c.id_link) for c in candidates.itertuples()
              if abs(c.link_length_m-p.link_length_m)<1e-5
              and pg.loc[p.id_link].hausdorff_distance(cg.loc[c.id_link])<1e-5
              and abs(pg.loc[p.id_link].length-cg.loc[c.id_link].length)<1e-5]
        if len(hits)!=1:missing.append(int(p.id_link))
        else:mapping[hits[0]]=int(p.id_link)
    if missing or len(mapping)!=len(parent)-len(set(selected)):
        raise ContractError(f'Exterior geometry/correspondence unresolved (including possible subdivisions): {missing}')
    return mapping


def node_heads(f):
    h=pd.concat([f[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),
                 f[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})])
    if (h.groupby('node')['head'].agg(lambda v:v.max()-v.min())>1e-9).any():
        raise ContractError('Parent shared-node heads inconsistent')
    return h.drop_duplicates('node').set_index('node')['head']


def correct(parent, child, mapping, selected, *, slope_floor=1e-8):
    """Freeze exterior rows; solve local heads and conservative positive reference flow.

    Absolute slope floor is explicit and checked on every replacement edge, including
    edges whose two ends are fixed. Reference flow fits saved child allocation subject
    to exact parent exterior fluxes; widths do not override those flux constraints.
    """
    if len(set(mapping.values()))!=len(mapping) or set(mapping.values())!=set(parent.id_link)-set(selected):
        raise ContractError('Exterior mapping is incomplete or noninjective')
    p=parent.set_index('id_link');f=child.set_index('id_link').copy()
    exterior=list(mapping);inside=f.index.difference(exterior);patch=f.loc[inside]
    if not len(inside):raise ContractError('No replacement edges')
    graph=nx.DiGraph();graph.add_edges_from(zip(f.u,f.v))
    if not nx.is_directed_acyclic_graph(graph):raise ContractError('Replacement has a directed cycle')
    selected_parent=parent[parent.id_link.isin(selected)]
    ext=parent[~parent.id_link.isin(selected)]
    nodes=set(patch.u)|set(patch.v)
    boundary=nodes & (set(ext.u)|set(ext.v))
    # Include network terminals even when no exterior reach witnesses them.
    boundary |= nodes & {v for v in graph if graph.in_degree(v)==0 or graph.out_degree(v)==0}
    ph=node_heads(parent)
    if not boundary<=set(ph.index):raise ContractError('Boundary lacks parent head')
    pg=nx.DiGraph();pg.add_edges_from(zip(selected_parent.u,selected_parent.v))
    cg=nx.DiGraph();cg.add_edges_from(zip(patch.u,patch.v))
    expected_boundary=(set(selected_parent.u)|set(selected_parent.v)) & (set(ext.u)|set(ext.v))
    if not expected_boundary<=boundary:raise ContractError('Selected boundary missing from replacement')
    connectivity_changes=[]
    for a in boundary:
        for b in boundary:
            if a!=b and nx.has_path(pg,a,b)!=nx.has_path(cg,a,b):
                connectivity_changes.append([int(a),int(b),bool(nx.has_path(pg,a,b)),bool(nx.has_path(cg,a,b))])
    free=sorted(nodes-boundary);ix={v:i for i,v in enumerate(free)}
    A=sparse.lil_matrix((len(patch),len(free)));fixed=np.zeros(len(patch))
    for j,r in enumerate(patch.itertuples()):
        for v,sign in [(r.u,1),(r.v,-1)]:
            if v in ix:A[j,ix[v]]+=sign
            else:fixed[j]+=sign*ph.loc[v]
    floor=slope_floor*patch.link_length_m.to_numpy();A=A.tocsr()
    if free:
        feasible=linprog(np.zeros(len(free)),A_ub=-A,b_ub=fixed-floor,bounds=[(None,None)]*len(free),method='highs')
        if not feasible.success:raise ContractError('Infeasible fixed boundary heads and positive slope floor: '+feasible.message)
        # Fit the old child slopes only inside the explicitly selected replacement.
        target=patch.slope.to_numpy()*patch.link_length_m.to_numpy()
        scale=np.maximum(target, floor)
        fun=lambda x:np.sum(((A@x+fixed-target)/scale)**2)
        jac=lambda x:np.asarray(2*A.T@((A@x+fixed-target)/scale**2)).ravel()
        solved=minimize(fun,feasible.x,jac=jac,constraints=[LinearConstraint(A.toarray(),floor-fixed,np.inf)],method='SLSQP',options={'maxiter':3000,'ftol':1e-12})
        if not solved.success:raise ContractError('Local head optimization failed: '+solved.message)
        heads=ph.copy()
        for v,h in zip(free,solved.x):heads.loc[v]=h
    else:heads=ph.copy()
    drops=patch.u.map(heads).to_numpy()-patch.v.map(heads).to_numpy()
    if np.any(drops<floor-1e-9):raise ContractError('Fixed-edge or solved slope floor infeasible')
    f.loc[inside,'head_us']=patch.u.map(heads);f.loc[inside,'head_ds']=patch.v.map(heads)
    f.loc[inside,'slope']=drops/patch.link_length_m
    # Node demands are the original selected subgraph divergence, including terminals.
    allnodes=sorted(set(patch.u)|set(patch.v));ni={v:i for i,v in enumerate(allnodes)}
    B=np.zeros((len(allnodes),len(patch)));d=np.zeros(len(allnodes))
    for r in selected_parent.itertuples():
        if r.u in boundary:d[ni[r.u]]+=r.reference_flow_fraction
        if r.v in boundary:d[ni[r.v]]-=r.reference_flow_fraction
    for j,r in enumerate(patch.itertuples()):B[ni[r.u],j]+=1;B[ni[r.v],j]-=1
    # Preserve gross incoming/outgoing port fluxes, not only their difference.
    # This also fixes exterior split fractions at mixed entry/exit boundaries.
    for v in sorted(boundary):
        for col in ["u","v"]:
            B=np.vstack([B,patch[col].eq(v).to_numpy(dtype=float)])
            d=np.r_[d,selected_parent.loc[selected_parent[col].eq(v),"reference_flow_fraction"].sum()]
    qmin=1e-12
    feasible=linprog(np.zeros(len(patch)),A_eq=B,b_eq=d,bounds=[(qmin,None)]*len(patch),method='highs')
    if not feasible.success:raise ContractError('Infeasible positive reference flow at frozen boundary ports: '+feasible.message)
    # L1 distance to prior allocation is linear and robust to redundant node equations.
    q0=patch.reference_flow_fraction.to_numpy();m=len(patch)
    solved=linprog(np.r_[np.zeros(m),np.ones(m)],A_ub=np.block([[np.eye(m),-np.eye(m)],[-np.eye(m),-np.eye(m)]]),b_ub=np.r_[q0,-q0],A_eq=np.c_[B,np.zeros_like(B)],b_eq=d,bounds=[(qmin,None)]*m+[(0,None)]*m,method='highs')
    if not solved.success:raise ContractError('Reference-flow fit failed')
    q=solved.x[:m]
    if (q<=0).any() or np.max(abs(B@q-d))>1e-9:raise ContractError('Reference flow conservation/positivity residual failed')
    f.loc[inside,'reference_flow_fraction']=q
    qref=float((parent.reference_discharge_cms/parent.reference_flow_fraction).median())
    f.loc[inside,'reference_discharge_cms']=q*qref
    # Inherit every exterior hydraulic/discretization field, even if widths were changed.
    fields=[c for c in parent.columns if c not in ['id_link','id_us_node','id_ds_node','u','v','geometry_id','geometry_release','hydraulic_head_policy']]
    f.loc[exterior,fields]=p.loc[[mapping[c] for c in exterior],fields].to_numpy()
    totals=f.groupby('u').reference_flow_fraction.sum()
    f.loc[inside,'split_fraction']=f.loc[inside,'reference_flow_fraction']/f.loc[inside,'u'].map(totals)
    # Fixed exterior split fractions must agree with the conserved new node totals.
    np.testing.assert_allclose(f.loc[exterior,'split_fraction'],f.loc[exterior,'reference_flow_fraction']/f.loc[exterior,'u'].map(totals),atol=1e-10,rtol=1e-10)
    local=f.loc[inside].copy();local['celerity_times_n']=5/3*local.depth_m**(2/3)*np.sqrt(local.slope)
    fast=np.clip(local.celerity_times_n/local.grid_n_min,local.celerity_min_mps,local.celerity_max_mps)
    ld=local.reference_discharge_cms/(local.width*local.slope*fast)
    local['count']=np.maximum(1,np.ceil(local.link_length_m/(local.cunge_alpha*ld))).astype(int)
    local['rapid_effective_length_m']=local.link_length_m/local['count']
    local=parameters(local,float(parent.n_manning.iloc[0]),beta=float(parent.diffusion_multiplier.iloc[0]))
    f.loc[inside,local.columns]=local
    f.loc[inside,'hydraulic_head_policy']='strict_selected_boundary_trial'
    pd.testing.assert_frame_equal(f.loc[exterior,fields].reset_index(drop=True),p.loc[[mapping[c] for c in exterior],fields].reset_index(drop=True),check_dtype=False)
    node_heads(f.reset_index())
    np.testing.assert_allclose((f.head_us-f.head_ds)/f.link_length_m,f.slope,rtol=1e-9,atol=1e-12)
    np.testing.assert_allclose(f.groupby('u').split_fraction.sum(),1.,atol=1e-10)
    for v in graph:
        if graph.in_degree(v) and graph.out_degree(v):
            incoming=f.loc[f.v.eq(v),'reference_flow_fraction'].sum()
            outgoing=f.loc[f.u.eq(v),'reference_flow_fraction'].sum()
            if abs(incoming-outgoing)>1e-9:raise ContractError('Full-network reference conservation failed')
    return f.reset_index(),dict(exterior_links=len(exterior),replacement_links=len(inside),boundary_nodes=sorted(map(int,boundary)),free_nodes=len(free),slope_floor=slope_floor,flow_residual=float(np.max(abs(B@q-d))),boundary_connectivity_changes=connectivity_changes)


def restore_named_exterior(parent, child, pg, cg, replacements):
    """Restore explicitly reviewed child->parent edges, then their shared port positions.

    No nearest-edge inference: require mapped directed endpoints to agree and unique
    parent positions. Only named exterior lines and incident replacement endpoints move.
    """
    from shapely.geometry import LineString,Point
    f=child.copy();geo=cg.copy();p=parent.set_index('id_link');pgeo=pg.set_index('id_link');old=cg.set_index('id_link')
    ports={};changes=[]
    for cid,pid in replacements.items():
        row=f[f.id_link.eq(cid)].iloc[0];pr=p.loc[pid]
        if (row.u,row.v)!=(pr.u,pr.v):raise ContractError('Explicit restoration endpoints disagree')
        # Orient parent geometry with upstream/downstream endpoint evidence from cg.
        line=pgeo.loc[pid].geometry;cl=old.loc[cid].geometry
        coords=list(line.coords)
        if Point(coords[0]).distance(Point(cl.coords[0]))>Point(coords[-1]).distance(Point(cl.coords[0])):coords.reverse()
        for v,pt in [(row.u,coords[0]),(row.v,coords[-1])]:
            if v in ports and ports[v]!=pt:raise ContractError('Inconsistent restored port position')
            ports[v]=pt
        geo.loc[geo.id_link.eq(cid),'geometry']=LineString(coords)
        f.loc[f.id_link.eq(cid),'link_length_m']=pr.link_length_m
        changes.append(dict(child=int(cid),parent=int(pid),action='restore_parent_exterior_geometry'))
    for idx,row in geo[~geo.id_link.isin(replacements)].iterrows():
        # These loaded geometry u/v identities are checked against the hydraulic frame.
        fr=f[f.id_link.eq(row.id_link)].iloc[0]
        if (row.u,row.v)!=(fr.u,fr.v):raise ContractError('Geometry/frame node identity mismatch')
        coords=list(row.geometry.coords);new=list(coords)
        if row.u in ports:new[0]=ports[row.u]
        if row.v in ports:new[-1]=ports[row.v]
        if new!=coords:
            geom=LineString(new);geo.at[idx,'geometry']=geom
            f.loc[f.id_link.eq(row.id_link),'link_length_m']=geom.length
            changes.append(dict(child=int(row.id_link),action='restore_shared_boundary_endpoint',old_length=row.geometry.length,new_length=geom.length))
    return f,geo,changes


def port_path_replacement(parent, parent_geometry, selected, *, max_paths=10000):
    """Experimental boundary-to-boundary path reduction, distinct from raster collapse.

    Retain every boundary port, stop paths at all ports (including mixed ports),
    preserve parent path probabilities. Parallel paths stay distinct. Width/depth
    use length-weighted means, a declared prototype policy, not new observations.
    """
    from shapely.geometry import LineString,Point
    import geopandas as gpd
    chosen=parent[parent.id_link.isin(selected)].copy();ext=parent[~parent.id_link.isin(selected)].copy()
    nodes=set(chosen.u)|set(chosen.v);g=nx.DiGraph();g.add_edges_from(zip(parent.u,parent.v))
    boundary=(nodes&(set(ext.u)|set(ext.v))) | (nodes&{v for v in g if g.in_degree(v)==0 or g.out_degree(v)==0})
    lookup=chosen.set_index('id_link');out={v:chosen.loc[chosen.u.eq(v),'id_link'].tolist() for v in nodes}
    pg=parent_geometry.set_index('id_link').geometry
    paths=[]
    def walk(path,q):
        v=lookup.loc[path[-1],'v']
        if v in boundary:
            paths.append((path,q))
            if len(paths)>max_paths:raise ContractError('Path-reduction size limit reached')
            return
        ids=out[v];total=lookup.loc[ids,'reference_flow_fraction'].sum()
        if not ids or total<=0:raise ContractError('Unresolved internal sink in selected graph')
        for lid in ids:walk(path+[lid],q*lookup.loc[lid,'reference_flow_fraction']/total)
    for v in sorted(boundary):
        for lid in out[v]:walk([lid],float(lookup.loc[lid,'reference_flow_fraction']))
    rows=[];geoms=[];lineage=[];next_id=int(parent.id_link.max())+1
    for path,q in paths:
        part=lookup.loc[path];row=part.iloc[0].copy();lid=next_id;next_id+=1
        row['id_link']=lid;row['u']=part.iloc[0].u;row['v']=part.iloc[-1].v
        row['id_us_node']=row.u;row['id_ds_node']=row.v
        row['link_length_m']=part.link_length_m.sum()
        for col in ['width','depth_m']:row[col]=np.average(part[col],weights=part.link_length_m)
        row['head_ds']=part.iloc[-1].head_ds;row['slope']=(row.head_us-row.head_ds)/row.link_length_m
        row['reference_flow_fraction']=q
        row['kb_local']=row.width/row.depth_m
        coords=[]
        for source in path:
            cc=list(pg.loc[source].coords)
            if coords:
                if Point(coords[-1]).distance(Point(cc[-1]))<Point(coords[-1]).distance(Point(cc[0])):cc.reverse()
                if Point(coords[-1]).distance(Point(cc[0]))>1e-5:raise ContractError('Parent path geometries do not join')
                coords.extend(cc[1:])
            else:
                # Source geometries are directed in the published parent release.
                coords=cc
        import hashlib
        row['geometry_id']=hashlib.sha256(LineString(coords).wkb).hexdigest()[:24]
        row['geometry_release']='isolated_port_path_trial_20261027'
        rows.append(row);geoms.append(dict(id_link=lid,u=int(row.u),v=int(row.v),geometry=LineString(coords)))
        lineage.append(dict(id_link=lid,parent_path=list(map(int,path)),reference_flow_fraction=q))
    f=pd.concat([ext,pd.DataFrame(rows)],ignore_index=True)
    geo=pd.concat([parent_geometry[parent_geometry.id_link.isin(ext.id_link)][['id_link','u','v','geometry']],gpd.GeoDataFrame(geoms,crs=parent_geometry.crs)],ignore_index=True)
    return f,gpd.GeoDataFrame(geo,geometry='geometry',crs=parent_geometry.crs),lineage
