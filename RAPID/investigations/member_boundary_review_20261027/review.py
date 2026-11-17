"""Reuse published compound-member contracts and test fixed-head slope feasibility.

Writes only a new investigation output directory. Never invokes production writers.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import numpy as np
import pandas as pd
import pyogrio
from network_variants.reviewed_geometry import ReviewedGeometry
from network_variants.boundary_contract_repair import derive_contract,ids
from feasibility import head_series,check_slopes
OUT=ROOT/'RAPID/outputs/member_boundary_review_20261027'
INV=ROOT/'RAPID/outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/records.parquet'
BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    global OUT
    prior=OUT
    retry='--retry' in sys.argv
    if retry:OUT=OUT.with_name(OUT.name+'_complete')
    OUT.mkdir(exist_ok=False)
    sources={};cache={};rows=[];boundary_rows=[];witnesses=[]
    if retry:
        sources=json.loads((prior/'manifest.json').read_text())['source_hashes']
        old=json.loads((prior/'state_results.json').read_text())
        retried={(str(v['example']).zfill(2),v['state']) for v in old if str(v.get('exception','')).startswith('TypeError:')}
        rows=[v for v in old if (str(v['example']).zfill(2),v['state']) not in retried]
        boundary_rows=[v for v in json.loads((prior/'boundary_nodes.json').read_text()) if (str(v['example']).zfill(2),v['state']) not in retried]
        witnesses=json.loads((prior/'feasibility_witnesses.json').read_text())
    def track(p):
        p=str(p)
        if p not in sources:sources[p]=digest(p)
        return Path(p)
    def readjson(p):return json.loads(track(p).read_text())
    def frame(p):return pd.read_parquet(track(p))
    inv=frame(INV);inv=inv[inv.edit_type.eq('Compound member')].copy();total_cases=len(inv)
    if retry:inv=inv[inv.apply(lambda z:(str(z.example).zfill(2),z.state) in retried,axis=1)]
    reviewed=ReviewedGeometry();track(reviewed.manifest)
    previous={}
    for path in (ROOT/'RAPID/outputs/boundary_contract_repair_20261008/full').glob('*/*/strict_head_failure.json'):
        previous[path.parent.parent.name,path.parent.name]=str(path)
    for num,z in enumerate(inv.itertuples(),1):
        ex=str(z.example);state=z.state;record=reviewed.lookup(ex,'indep',state)
        out=dict(example=ex,state=state,previous_strict_failure=(ex,state) in previous)
        try:
            if ex not in cache:
                br=reviewed.lookup(ex,'indep','S000_base');parent=frame(br['original_frame']);heads=head_series(parent)
                positions=frame(BASE/ex/'floor_0.05/old/nodes.parquet').set_index('id_node').reference_section_dist_out_m
                membership=pd.read_csv(track(Path(br['links']).parents[2]/'variant/summary/unit_link_membership.csv'))
                bn=pyogrio.read_dataframe(track(br['nodes']),columns=['id_node']);bp=bn.set_index('id_node').geometry
                # Map raw node IDs to canonical identities, just as current loader does.
                pairs=pd.concat([parent[['id_us_node','u']].rename(columns={'id_us_node':'raw','u':'node'}),parent[['id_ds_node','v']].rename(columns={'id_ds_node':'raw','v':'node'})]).drop_duplicates()
                bp.index=bp.index.map(pairs.set_index('raw').node)
                cache[ex]=parent,heads,positions,membership,bp
            parent,heads,positions,membership,bp=cache[ex]
            contract=readjson(record['contract_path']);child=frame(record['original_frame'])
            if digest(record['contract_path'])!=record['contract_sha256']:raise ValueError('Contract checksum differs from published record')
            for path in [record['original_frame'],record['nodes'],record['links']]:
                track(path);expected=record.get('retained_source_checksums',{}).get(path)
                if expected and sources[path]!=expected:raise ValueError('Published geometry/frame checksum mismatch: '+path)
            selected=set(map(int,contract['selected_parent_links']));unit_ids=set(ids(contract['selected_unit_ids']))
            saved_unit_ids=set(map(int,z.selected_unit_ids))
            member=set(membership[membership.unit_id.isin(unit_ids)].id_link.astype(int))
            if not contract.get('valid') or not contract.get('target_ok'):raise ValueError('Saved contract/target not valid')
            if unit_ids!=saved_unit_ids or member!=selected:raise ValueError('Saved contract differs from existing unit selection/membership')
            # Cross-check the saved boundary, never replace it with a newly inferred scope.
            cross=derive_contract(parent.rename(columns={'id_us_node':'raw_us','id_ds_node':'raw_ds','u':'id_us_node','v':'id_ds_node'}),selected)
            if set(cross['boundary_nodes'])!=set(contract['boundary_nodes']):raise ValueError('Saved boundary and selected/exterior incidence disagree')
            out.update(contract_valid=True,selected_links=len(selected),saved_boundaries=len(contract['boundary_nodes']),decision=record.get('decision'),historical_boundary_status=record.get('historical_automatic_status'),contract_path=record['contract_path'])
            if any(k.startswith('reviewed_') for k in contract):
                out.update(boundary_status='manual_boundary_exception',feasibility_status='not_tested',exception='Saved manual boundary interpretation retained for separate review');rows.append(out);continue
            pairs=pd.concat([child[['id_us_node','u']].rename(columns={'id_us_node':'raw','u':'node'}),child[['id_ds_node','v']].rename(columns={'id_ds_node':'raw','v':'node'})]).drop_duplicates()
            if pairs.raw.duplicated().any() or pairs.node.duplicated().any():raise ValueError('Current node mapping is not injective')
            mapping=dict(zip(pairs.raw.astype(int),pairs.node.astype(int)));inverse={v:k for k,v in mapping.items()}
            explicit={int(k):int(v) for k,v in readjson(record['validated_correspondences_path']).items()} if record.get('validated_correspondences_path') else {}
            if any(mapping.get(k)!=v for k,v in explicit.items()):raise ValueError('Published validated correspondence disagrees with current frame')
            nodes=pyogrio.read_dataframe(record['nodes'],columns=['id_node']).set_index('id_node')
            boundaries=set(contract['boundary_nodes']);missing=boundaries-set(inverse)
            displaced=[];unverified=[]
            for v in sorted(boundaries):
                raw=inverse.get(v);distance=float(nodes.loc[raw].geometry.distance(bp.loc[v])) if raw is not None else None
                proof='exact_position' if distance is not None and distance<=1e-5 else 'validated_displaced_identity' if raw in explicit and explicit[raw]==v else 'missing' if raw is None else 'unverified_displaced_identity'
                if proof=='validated_displaced_identity':displaced.append(v)
                if proof=='unverified_displaced_identity':unverified.append(v)
                spec=next(b for b in contract['boundaries'] if b['parent_node']==v)
                boundary_rows.append(dict(example=ex,state=state,parent_node=int(v),child_raw_node=raw,reference_head_m=float(heads.loc[v]),displacement_m=distance,correspondence=proof,exterior_in=spec['exterior_in'],exterior_out=spec['exterior_out']))
            out.update(missing_boundary_nodes=sorted(map(int,missing)),validated_displaced_boundaries=len(displaced),unverified_boundaries=unverified)
            if missing or unverified:
                out.update(boundary_status='correspondence_exception',feasibility_status='not_tested');rows.append(out);continue
            # Reuse saved exterior paths; on unchanged cases reconstruct unique single
            # edge correspondences from the already mapped endpoints and length.
            ext=parent[~parent.id_link.isin(selected)];paths=[]
            pathfile=Path(record.get('source_bundle') or '')/'exterior_paths.json'
            if record.get('source_bundle') and pathfile.exists():paths=readjson(pathfile)
            else:
                groups={(u,v):g for (u,v),g in child.groupby(['u','v'])}
                for p in ext.itertuples():
                    candidates=groups.get((p.u,p.v),child.iloc[:0])
                    candidates=candidates[np.isclose(candidates.link_length_m,p.link_length_m,rtol=1e-8,atol=1e-5)]
                    if len(candidates)>1 and p.id_link in set(candidates.id_link):candidates=candidates[candidates.id_link.eq(p.id_link)]
                    paths.append(dict(parent_link=int(p.id_link),child_links=candidates.id_link.astype(int).tolist(),path_unique=len(candidates)==1))
            lookup=child.set_index('id_link');parent_lookup=parent.set_index('id_link');outside_edges=set();extra_nodes=set();length_changes=0;path_errors=[]
            if {p['parent_link'] for p in paths}!=set(ext.id_link):path_errors.append('saved exterior coverage differs')
            for path in paths:
                lid=path['parent_link'];chain=path['child_links'];pr=parent_lookup.loc[lid]
                if not path.get('path_unique',True) or not chain or not set(chain)<=set(lookup.index):path_errors.append(str(lid));continue
                current=pr.u
                for c in chain:
                    cr=lookup.loc[c]
                    if cr.u!=current or c in outside_edges:path_errors.append(str(lid));break
                    outside_edges.add(c);current=cr.v
                if current!=pr.v:path_errors.append(str(lid))
                chain_nodes=set(lookup.loc[chain].u)|set(lookup.loc[chain].v)
                extra_nodes|=chain_nodes-{pr.u,pr.v}
                if not np.isclose(lookup.loc[chain].link_length_m.sum(),pr.link_length_m,rtol=1e-8,atol=1e-5):length_changes+=1
            out.update(boundary_status='verified_existing_correspondences',exterior_length_changes=length_changes,exterior_path_errors=path_errors,exterior_subdivision_nodes=sorted(map(int,extra_nodes)))
            if path_errors or extra_nodes:
                out.update(feasibility_status='not_tested_exterior_path_exception');rows.append(out);continue
            ext_nodes=set(ext.u)|set(ext.v)
            if not ext_nodes<=set(inverse):raise ValueError('Exterior parent node missing from current mapping')
            patch=child[~child.id_link.isin(outside_edges)]
            if not boundaries<=(set(patch.u)|set(patch.v)):
                out.update(feasibility_status='not_tested_missing_replacement_incidence');rows.append(out);continue
            fixed=ext_nodes|boundaries
            result=check_slopes(patch,heads,fixed,positions)
            out.update(feasibility_status=result['status'],replacement_links=len(patch),free_nodes=result['free_nodes'],positive_slopes_feasible=result['positive_slopes_feasible'],current_relative_floor_feasible=result['current_relative_floor_feasible'],maximum_uniform_positive_slope=result['maximum_uniform_positive_slope'],maximum_floor_fraction=result['maximum_floor_fraction'])
            if result['status']!='feasible' or (ex,state) in [('22','S296_unit_30'),('22','S001_unit_52')]:witnesses.append(dict(example=ex,state=state,**result))
            rows.append(out)
        except Exception as e:
            out.update(boundary_status='review_exception',feasibility_status='not_tested',exception=f'{type(e).__name__}: {e}');rows.append(out)
        if num%100==0:
            print(num,'/',len(inv),pd.Series([v.get('feasibility_status') for v in rows]).value_counts().to_dict(),flush=True)
            pd.DataFrame(rows).to_json(OUT/'progress.json',orient='records',indent=2)
    pd.DataFrame(rows).to_json(OUT/'state_results.json',orient='records',indent=2)
    pd.DataFrame(rows).to_csv(OUT/'state_results.csv',index=False)
    pd.DataFrame(boundary_rows).to_json(OUT/'boundary_nodes.json',orient='records',indent=2)
    (OUT/'feasibility_witnesses.json').write_text(json.dumps(witnesses,indent=2))
    for path,h in sources.items():
        if digest(path)!=h:raise RuntimeError('Source changed while reading: '+path)
    summary=dict(compound_member_cases=total_cases,geometry_release=reviewed.release,boundary_status=pd.Series([v.get('boundary_status') for v in rows]).value_counts().to_dict(),feasibility_status=pd.Series([v.get('feasibility_status') for v in rows]).value_counts().to_dict(),previous_strict_failure_cases=sum(v['previous_strict_failure'] for v in rows),source_files=len(sources),production_modified=False,slope_optimization_runs=0,routing_runs=0,source_hashes=sources)
    (OUT/'manifest.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k!='source_hashes'},flush=True)

if __name__=='__main__':run()
