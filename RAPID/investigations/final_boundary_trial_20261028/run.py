"""Accepted isolated geometry/head/slope trial. No routing or production writes."""
from pathlib import Path
import sys,json,hashlib,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/member_boundary_review_20261027'),str(ROOT/'RAPID/investigations/boundary_objectives_20261027'),str(ROOT/'RAPID/investigations/accepted_boundary_trial_20261027')]
import numpy as np,pandas as pd,geopandas as gpd,pyogrio,networkx as nx
from shapely.geometry import LineString,Point
from feasibility import head_series,path_constraints
from solver import quadratic_projection
from solve import fit_internal,fit_primary
from network_variants.reviewed_geometry import ReviewedGeometry
from network_variants.boundary_contract_repair import derive_contract
OUT=ROOT/'RAPID/outputs/final_boundary_trial_20261028';BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';release=ReviewedGeometry();sources={}
def dumps(value,**kwargs):return json.dumps(value,default=lambda x:x.item() if isinstance(x,np.generic) else str(x),**kwargs)
def track(p):
 p=Path(p);sources.setdefault(str(p),hashlib.sha256(p.read_bytes()).hexdigest());return p
def js(p):return json.loads(track(p).read_text())
def read(p):return pd.read_parquet(track(p))
def vectors(p):return pyogrio.read_dataframe(track(p))
def pts(f,n):
 m=dict(zip(f.u,f.id_us_node))|dict(zip(f.v,f.id_ds_node));ns=n.set_index('id_node');return {int(v):ns.loc[raw].geometry for v,raw in m.items()}
def sig(f):
 g=nx.MultiDiGraph();g.add_edges_from(zip(f.u,f.v));return dict(dag=nx.is_directed_acyclic_graph(g),components=nx.number_weakly_connected_components(g),sources=sorted(int(v) for v in g if g.in_degree(v)==0),sinks=sorted(int(v) for v in g if g.out_degree(v)==0),cycle_rank=g.number_of_edges()-g.number_of_nodes()+nx.number_weakly_connected_components(g))
def orient(line,up):
 assert line.geom_type=='LineString'
 return line if Point(line.coords[0]).distance(up)<=Point(line.coords[-1]).distance(up) else LineString(list(line.coords)[::-1])
def main():
 OUT.mkdir(exist_ok=False);track(release.manifest)
 decisions=js('/Users/6256481/Downloads/boundary-fix-decisions-after-geometry.json');assert len(decisions['cases'])==27 and all(c['decision']=='Accept recommended fix for isolated testing' for c in decisions['cases'])
 proposals={v['key']:v for v in js(ROOT/'RAPID/outputs/proposed_boundary_fixes_20261027/proposals.json')};manifest=js(ROOT/'RAPID/outputs/proposed_boundary_fixes_20261027/review_manifest.json')
 assert all(c['proposal_id']==manifest['proposal_ids'][c['key']] for c in decisions['cases'])
 (OUT/'accepted_decisions.json').write_text(dumps(decisions,indent=2))
 residual=js(ROOT/'RAPID/outputs/residual_decision_check_20261028/accepted_review.json')
 latest=js('/Users/6256481/Downloads/residual-exterior-geometry-decisions (1).json');assert residual==latest
 residual_by_case={v['key']:v['reaches'] for v in residual['cases']}
 assert sum(v['decision']=='Accept restoration trial' for c in residual['cases'] for v in c['reaches'])==14
 assert sum(v['decision']=='Keep current geometry as exception' for c in residual['cases'] for v in c['reaches'])==20
 (OUT/'residual_decisions.json').write_text(dumps(residual,indent=2))
 (OUT/'unit143_acceptance.json').write_text(dumps(dict(accepted_in_conversation=True,scope='Retain reversal of exterior parent 434/current 979 and allow necessary boundary flexibility; no automatic approval of other geometry differences.'),indent=2))

 cases=[(c['key'],'accepted_exception') for c in decisions['cases']]+[(f'22/{s}','control') for s in ['S296_unit_30','S023_unit_136','S166_unit_309','S001_unit_52','S013_unit_8']]
 cache={};summaries=[];all_edges=[];all_nodes=[]
 for key,kind in cases:
  ex,state=key.split('/');dest=OUT/ex/state;dest.mkdir(parents=True);summary=dict(key=key,kind=kind)
  try:
   if ex not in cache:
    br=release.lookup(ex,'indep','S000_base');p=read(br['original_frame']);pg=vectors(br['links']).set_index('id_link');pn=vectors(br['nodes']);p=p.merge(pg[['geometry']],left_on='id_link',right_index=True,validate='one_to_one');pp=pts(p,pn);old=read(BASE/ex/'floor_0.05/old/links.parquet').set_index('id_link');cache[ex]=p,pp,old
   p,pp,old=cache[ex];pi=p.set_index('id_link');refheads=head_series(p);r=release.lookup(ex,'indep',state);original=read(r['original_frame']);cg=vectors(r['links']).set_index('id_link');ns=vectors(r['nodes']);f=original.merge(cg[['geometry']],left_on='id_link',right_index=True,validate='one_to_one');cp=pts(f,ns);contract=js(r['contract_path']);selected=set(contract['selected_parent_links']);scope=set(selected);actions=[];reversed_map={};restored={}
   for i,q in f.iterrows():f.at[i,'geometry']=orient(q.geometry,cp[q.u])
   proposal=proposals.get(key);edit=proposal.get('graph_check') if proposal else None
   if edit:
    remove=set(edit['removed_child_links']);restore=edit['restored_parent_links'];f=f[~f.id_link.isin(remove)].copy()
    for lid in restore:
     row=pi.loc[lid].copy();row['id_link']=-int(lid);row['geometry']=orient(row.geometry,pp[row.u]);f=pd.concat([f,pd.DataFrame([row])],ignore_index=True);restored[-lid]=lid
     for v in [int(row.u),int(row.v)]:cp[v]=pp[v]
    actions.append(dict(action='accepted_reference_reach_restoration',removed_child_links=sorted(remove),restored_parent_links=restore))
    # Enforce shared coordinates at restored boundaries, changing only incident remaining endpoints.
    for i,q in f.iterrows():
     coords=list(q.geometry.coords);new=list(coords);new[0]=(cp[q.u].x,cp[q.u].y);new[-1]=(cp[q.v].x,cp[q.v].y)
     if new!=coords:
      f.at[i,'geometry']=LineString(new);actions.append(dict(action='reattach_to_restored_boundary',child_link=int(q.id_link),old_length=float(q.link_length_m),new_length=float(f.at[i,'geometry'].length)))
   if key=='22/S262_unit_287':scope.add(864);actions.append(dict(action='accepted_scope_expansion',parent_link=864))
   if key=='22/S244_unit_262':reversed_map={998:783}
   if key=='37/S252_unit_201':reversed_map={977:600}
   if key=='37/S289_unit_143':reversed_map={979:434}
   # Apply exactly the latest per-reach choices, retaining existing child IDs.
   chosen=residual_by_case.get(key,[]);approved_restore=[z for z in chosen if z['decision']=='Accept restoration trial'];kept={z['child_link']:z for z in chosen if z['decision']=='Keep current geometry as exception'}
   before_residual=f.copy();before_index=before_residual.set_index('id_link');moved=set();restore_ids=set()
   for z in approved_restore:
    lid=z['child_link'];pr=pi.loc[z['parent_link']];idx=f.index[f.id_link.eq(lid)];assert len(idx)==1
    assert int(f.loc[idx[0],'u'])==int(pr.u) and int(f.loc[idx[0],'v'])==int(pr.v)
    f.at[idx[0],'geometry']=orient(pr.geometry,pp[pr.u]);restore_ids.add(lid)
    for node in z['moved_endpoints']:moved.add(node['node']);cp[node['node']]=pp[node['node']]
    actions.append(dict(action='accepted_residual_restoration',parent_link=z['parent_link'],child_link=lid,moved_nodes=[n['node'] for n in z['moved_endpoints']]))
   for i,q in f.iterrows():
    if q.id_link in restore_ids or not (q.u in moved or q.v in moved):continue
    assert q.id_link not in kept,'Restoration would move an explicitly retained exterior reach'
    coords=list(q.geometry.coords);coords[0]=(cp[q.u].x,cp[q.u].y);coords[-1]=(cp[q.v].x,cp[q.v].y);f.at[i,'geometry']=LineString(coords)
    actions.append(dict(action='residual_selected_side_reattachment',child_link=int(q.id_link),old_length=float(q.geometry.length),new_length=float(f.at[i,'geometry'].length)))
   after_index=f.set_index('id_link')
   for lid in kept:assert after_index.loc[lid].geometry.equals_exact(before_index.loc[lid].geometry,1e-9)
   gpd.GeoDataFrame(before_residual[['id_link','u','v','geometry']],geometry='geometry',crs=cg.crs).to_parquet(dest/'before_residual_geometry.parquet',index=False)
   summary.update(approved_residual_restorations=len(approved_restore),kept_geometry_exceptions=len(kept))

   f=f.reset_index(drop=True);f['link_length_m']=f.geometry.map(lambda g:g.length)
   f['u']=f.u.astype(int);f['v']=f.v.astype(int);f['id_link']=f.id_link.astype(int)
   assert (f.link_length_m>0).all();assert sig(original)==sig(f),'Graph signature changed'
   gb=nx.DiGraph();gb.add_edges_from(zip(original.u,original.v));ga=nx.DiGraph();ga.add_edges_from(zip(f.u,f.v));common=set(gb)&set(ga);assert all((nx.descendants(gb,v)&common)==(nx.descendants(ga,v)&common) for v in common)
   endpoint_error=max(max(Point(q.geometry.coords[0]).distance(cp[q.u]),Point(q.geometry.coords[-1]).distance(cp[q.v])) for q in f.itertuples());assert endpoint_error<1e-6
   boundary=derive_contract(p.rename(columns={'id_us_node':'raw_us','id_ds_node':'raw_ds','u':'id_us_node','v':'id_ds_node'}),scope)['boundary_nodes'];outside={};exterior_evidence=[];unmatched=[];fi=f.set_index('id_link')
   # Reuse reviewed evidence for retained one-edge exterior correspondences, but validate current direction.
   saved={};pathfile=Path(r.get('source_bundle') or '')/'exterior_paths.json'
   if r.get('source_bundle') and pathfile.exists():
    for z in js(pathfile):
     if z.get('path_unique') and len(z['child_links'])==1:saved[z['parent_link']]=z['child_links'][0]
   groups={(int(u),int(v)):g for (u,v),g in f.groupby(['u','v'])}
   for q in p[~p.id_link.isin(scope)].itertuples():
    lid=int(q.id_link)
    if lid in reversed_map.values():cid=next(c for c,v in reversed_map.items() if v==lid);role='reversed_exterior'
    elif -lid in restored:cid=-lid;role='exterior'
    else:
     hits=groups.get((q.u,q.v),f.iloc[:0]);cid=saved.get(lid)
     if cid not in set(hits.id_link):
      if len(hits)==1:cid=int(hits.id_link.iloc[0])
      elif len(hits)>1:
       distances=hits.geometry.map(q.geometry.hausdorff_distance);exact=hits[distances<1e-6];cid=int(exact.id_link.iloc[0]) if len(exact)==1 else None
      else:cid=None
     role='exterior'
    if cid is None:unmatched.append(lid);continue
    assert cid not in outside,'Exterior edge reused';outside[cid]=lid;z=fi.loc[cid];distance=q.geometry.hausdorff_distance(z.geometry);length_delta=float(z.link_length_m-q.link_length_m)
    exterior_evidence.append(dict(parent_link=lid,child_link=cid,role=role,length_delta_m=length_delta,hausdorff_m=float(distance),exact_geometry=abs(length_delta)<1e-6 and distance<1e-6))
   (dest/'geometry_actions.json').write_text(dumps(actions,indent=2));(dest/'exterior_correspondences.json').write_text(dumps(exterior_evidence,indent=2));(dest/'boundary_contract.json').write_text(dumps(dict(original_selected=sorted(selected),approved_scope=sorted(scope),boundary_nodes=boundary,accepted_reversed_connectors=reversed_map,unmatched_exterior_parent_links=unmatched),indent=2))
   f['parent_link']=f.id_link.map(outside);f['role']=f.id_link.map(lambda lid:'reversed_exterior' if lid in reversed_map else 'exterior' if lid in outside else 'selected_replacement')
   gpd.GeoDataFrame(f[['id_link','u','v','link_length_m','parent_link','role','geometry']],geometry='geometry',crs=cg.crs).to_parquet(dest/'geometry_candidate.parquet',index=False)
   summary.update(geometry_checks_pass=True,endpoint_error_m=endpoint_error,unmatched_exterior=len(unmatched),exterior_geometry_changes=sum(not z['exact_geometry'] for z in exterior_evidence),selected_links=len(selected),scope_links=len(scope),restored_parent_links=len(restored))
   if unmatched:raise ValueError('Unmatched exterior reaches: '+str(unmatched))
   patch=f[f.role.eq('selected_replacement')].copy();floorvals=old.loc[list(scope),'lower_bound'].dropna().unique();assert len(floorvals)==1,('Selected region spans incompatible floors',floorvals)
   floor=float(floorvals[0]);sref=floor/.01;ext=p[~p.id_link.isin(scope)];fixed=set(ext.u)|set(ext.v)|set(boundary);nodes=set(f.u)|set(f.v);assert fixed<=nodes
   floors=[]
   for q in f.itertuples():
    if q.id_link in outside:
     lid=outside[q.id_link];low=old.loc[lid,'lower_bound'];floors.append(min(low,pi.loc[lid,'slope']) if np.isfinite(low) else 0.)
    else:floors.append(floor)
   drops=np.maximum(np.array(floors)*f.link_length_m.to_numpy(),1e-10)
   paths=path_constraints(f,refheads.loc[sorted(fixed)],drops);bad=[z for z in paths if z['available_drop_m']<z['path_weight']-1e-9]
   candidates=sorted({v for z in bad for v in [z['upstream'],z['downstream']]});allowed=set(boundary)|{v for cid in reversed_map for v in [int(fi.loc[cid].u),int(fi.loc[cid].v)]}
   if not set(candidates)<=allowed:raise ValueError('Floor conflict requires unapproved exterior-node release: '+str(set(candidates)-allowed))
   newfixed=refheads.loc[sorted(fixed)].to_dict();primary=dict(candidates=candidates,strict_fixed_feasible=not bad,conflicts=bad)
   if candidates:
    ix={v:i for i,v in enumerate(candidates)};A=[];b=[]
    for z in paths:
     row=np.zeros(len(candidates))
     for v,sgn in [(z['upstream'],1),(z['downstream'],-1)]:
      if v in ix:row[ix[v]]+=sgn
     rhs=z['path_weight']-z['available_drop_m']
     if np.any(row):A.append(row);b.append(rhs)
     elif rhs>1e-8:raise ValueError('Unrelated exterior floor conflict')
    affected=f[f.role.eq('exterior')&(f.u.isin(candidates)|f.v.isin(candidates))];C=np.zeros((len(affected),len(candidates)));D=[]
    for j,q in enumerate(affected.itertuples()):
     for v,sgn in [(q.u,1),(q.v,-1)]:
      if v in ix:C[j,ix[v]]+=sgn
     pr=pi.loc[outside[q.id_link]];D.append(pr.head_us-pr.head_ds)
    scale=max(max(b),.001);delta,qa=fit_primary(C,1/np.array(D),np.array(A),np.array(b)/scale);delta*=scale
    for v,d in zip(candidates,delta):newfixed[v]+=float(d)
    primary.update(head_changes={int(v):float(d) for v,d in zip(candidates,delta)},certificate=qa,objective=float(np.sum((C@delta/np.array(D))**2)))
   (dest/'primary_head_fit.json').write_text(dumps(primary,indent=2))
   # Fix optimal boundary heads before the secondary fit, which only involves replacement edges.
   patchfixed={v:newfixed[v] for v in (set(patch.u)|set(patch.v))&set(newfixed)}
   heads,secondary=fit_internal(patch,patchfixed,sref,np.maximum(patch.link_length_m.to_numpy()*floor,1e-10));heads.update(newfixed)
   f['head_us']=f.u.map(heads);f['head_ds']=f.v.map(heads);assert not f.head_us.isna().any() and not f.head_ds.isna().any();f['slope']=(f.head_us-f.head_ds)/f.link_length_m;f['floor']=floors
   deficit=drops-(f.head_us-f.head_ds).to_numpy();assert max(deficit)<1e-8 and (f.slope>0).all()
   (dest/'secondary_head_fit.json').write_text(dumps(secondary,indent=2))
   # Publish only investigative geometry/head/slope fields: no stale width/depth/routing coefficients.
   gpd.GeoDataFrame(f[['id_link','u','v','link_length_m','head_us','head_ds','slope','floor','parent_link','role','geometry']],geometry='geometry',crs=cg.crs).to_parquet(dest/'slope_candidate.parquet',index=False)
   for q in f[f.id_link.isin(outside)].itertuples():
    pr=pi.loc[outside[q.id_link]];prior=original[original.id_link.eq(q.id_link)];saved=float(prior.slope.iloc[0]) if len(prior) else None
    all_edges.append(dict(key=key,parent_link=int(outside[q.id_link]),child_link=int(q.id_link),role=q.role,reference_slope=float(pr.slope),saved_slope=saved,new_slope=float(q.slope),relative_slope_change=float(q.slope/pr.slope-1),reference_length_m=float(pr.link_length_m),new_length_m=float(q.link_length_m),head_us_change=float(q.head_us-refheads.loc[q.u]),head_ds_change=float(q.head_ds-refheads.loc[q.v])))
   for v,h in heads.items():
    all_nodes.append(dict(key=key,node=int(v),head=float(h),reference_head=float(refheads.loc[v]) if v in refheads.index else None,fixed_exterior=v in fixed,released=v in candidates))
   this=[v for v in all_edges if v['key']==key];summary.update(status='solved',fixed_boundary_feasible=not bad,released_nodes=candidates,max_floor_deficit_m=max(0.,float(max(deficit))),secondary_stationarity=secondary['stationarity'],changed_exterior_slopes=sum(abs(z['relative_slope_change'])>1e-8 for z in this),max_relative_exterior_slope_change=max(abs(z['relative_slope_change']) for z in this),changed_exterior_heads=sum(abs(heads[v]-refheads.loc[v])>1e-8 for v in fixed),ready_for_strict_scope_review=not any(not z['exact_geometry'] for z in exterior_evidence))
  except Exception as e:
   summary.update(status='blocked_explicitly',reason=str(e));(dest/'failure.txt').write_text(traceback.format_exc())
  summaries.append(summary);print(key,summary['status'],summary.get('reason',''),flush=True)
 (OUT/'case_results.json').write_text(dumps(summaries,indent=2));pd.DataFrame(summaries).to_csv(OUT/'case_results.csv',index=False);pd.DataFrame(all_edges).to_parquet(OUT/'exterior_slope_comparison.parquet',index=False);pd.DataFrame(all_nodes).to_parquet(OUT/'node_heads.parquet',index=False)
 for path,h in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 (OUT/'manifest.json').write_text(dumps(dict(source_hashes=sources,production_modified=False,routing_runs=0,rf_training_runs=0,cases=len(cases),solved=sum(z['status']=='solved' for z in summaries)),indent=2))
if __name__=='__main__':main()
