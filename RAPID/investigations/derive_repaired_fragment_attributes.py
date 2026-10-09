"""Isolated fragment width/head checks; not Cunge preparation or a routing run.

Exterior widths are inherited only where whole parent geometry is recovered.
New interior fragment samples are re-trimmed, never assigned the merged width.
The learned width-correction policy and calibrated K/X must still be regenerated.
"""
from pathlib import Path
import sys,json,argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from network_variants.width_trim_selection import select_width_trim
from network_variants.shared_head_reconstruction import reconstruct_shared_heads
BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
OUT=ROOT/'RAPID/outputs/boundary_contract_repair_20261008'


def derive(row,folder):
    geometry=gpd.read_parquet(folder/'trial_links.parquet')
    old=pd.read_parquet(row.frame_path).set_index('id_link');parent=pd.read_parquet(Path(row.frame_path).with_name(row.parent_fingerprint+'.parquet')).set_index('id_link')
    contract=json.loads((folder/'contract.json').read_text());validation=json.loads((folder/'validation.json').read_text())
    exact={lid:r['parent_link'] for r in validation['exterior'] if r['geometry_exact'] for lid in r['child_links']}
    split=set(geometry.groupby('source_link').size().loc[lambda s:s>1].index)
    need=geometry[geometry.source_link.isin(split)&~geometry.id_link.isin(exact)]
    records=[];sample_groups={}
    if len(need):
        widthdir=Path(row.variant_manifest).parents[1]/'widths'
        parquet=widthdir/'link_width_samples.parquet';csv=widthdir/'link_width_samples.csv'
        if parquet.exists():samples=pd.read_parquet(parquet,filters=[('id_link','in',list(need.source_link.unique()))])
        elif csv.exists():
            cols=['id_link','sample_index','x','y','width_total','width_wet','width_dry','dt_width_global']
            parts=[]
            for chunk in pd.read_csv(csv,usecols=lambda c:c in cols,chunksize=100000):
                part=chunk[chunk.id_link.isin(need.source_link)]
                if len(part):parts.append(part)
            samples=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        else:samples=pd.DataFrame()
        for source in need.source_link.unique():
            group=samples[samples.id_link.eq(source)].copy() if len(samples) else pd.DataFrame()
            if group.empty:continue
            fragments=geometry[geometry.source_link.eq(source)]
            # Source fragments form a directed chain; order using their endpoints.
            pieces={int(r.id_us_node):r for r in fragments.itertuples()}
            start=next(u for u in pieces if u not in set(fragments.id_ds_node));ordered=[]
            while start in pieces:r=pieces[start];ordered.append(r);start=int(r.id_ds_node)
            coords=[]
            for r in ordered:coords.extend(list(r.geometry.coords) if not coords else list(r.geometry.coords)[1:])
            from shapely.geometry import LineString
            line=LineString(coords)
            group['_distance']=[line.project(Point(x,y)) for x,y in zip(group.x,group.y)]
            cumulative=0.
            for i,r in enumerate(ordered):
                end=cumulative+r.geometry.length
                subset=group[(group._distance>=cumulative-1e-6)&((group._distance<end-1e-6) if i<len(ordered)-1 else (group._distance<=end+1e-6))].sort_values('_distance').copy()
                subset['sample_index']=range(len(subset));sample_groups[int(r.id_link)]=subset;cumulative=end
    for r in geometry.itertuples():
        source=old.loc[r.source_link];status='saved_unsplit_width';width=float(source.width);trim='';count=0
        if r.id_link in exact:width=float(parent.loc[exact[r.id_link]].width);status='frozen_parent_exterior_width'
        elif r.source_link in split:
            group=sample_groups.get(r.id_link,pd.DataFrame());width=np.nan;status='missing_fragment_samples'
            if len(group):
                choice=select_width_trim(pd.Series(dtype=object),group);trim=choice.method;count=choice.stop-choice.start
                if choice.valid_core:width=float(group.iloc[choice.start:choice.stop].width_wet.mean());status='retrimmed_fragment_wet_width_pending_QC'
                else:status='fragment_has_no_valid_trim_core'
        records.append(dict(id_link=int(r.id_link),source_link=int(r.source_link),u=int(r.u),v=int(r.v),link_length_m=float(r.geometry.length),width=width,previous_merged_width=float(source.width),width_status=status,trim_method=trim,selected_samples=count))
    f=pd.DataFrame(records);f.to_csv(folder/'fragment_width_audit.csv',index=False)
    refs=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example').loc[row.example]
    f['depth_m']=refs.stored_depth_m*(f.width/refs.reference_width_m)**.65
    nodes=pd.read_parquet(BASE/row.example/'floor_0.05/old/nodes.parquet')
    base_links=pd.read_parquet(BASE/row.example/'floor_0.05/old/links.parquet')
    exterior=base_links[~base_links.id_link.isin(contract['selected_parent_links'])]
    fixed=set(exterior.id_us_node)|set(exterior.id_ds_node)|set(contract['boundary_nodes'])
    reference_heads=nodes.set_index('id_node')['head']
    try:
        f,heads_qa=reconstruct_shared_heads(f,base_links,nodes,reference_heads,fixed_node_ids=fixed)
    except ValueError as error:
        diagnostic=dict(strict_fixed_exterior_failure=str(error),routing_ready=False)
        try:
            relaxed,relaxed_qa=reconstruct_shared_heads(f,base_links,nodes,reference_heads)
            inferred=pd.concat([relaxed[['u','head_us']].rename(columns={'u':'node','head_us':'head'}),relaxed[['v','head_ds']].rename(columns={'v':'node','head_ds':'head'})]).drop_duplicates('node').set_index('node')['head']
            compare=sorted(fixed&set(inferred.index))
            differences=(inferred.loc[compare]-reference_heads.loc[compare]).abs()
            diagnostic.update(current_head_policy_passed=True,changed_exterior_heads=int(differences.gt(1e-8).sum()),maximum_exterior_head_change_m=float(differences.max()),current_policy_head_qa=relaxed_qa)
            relaxed.to_parquet(folder/'diagnostic_relaxed_heads.parquet',index=False)
        except ValueError as second:
            diagnostic.update(current_head_policy_passed=False,current_policy_error=str(second))
        (folder/'strict_head_failure.json').write_text(json.dumps(diagnostic,indent=2))
        return dict(example=row.example,state=row.state,status='blocked',reason=str(error),widths_valid=bool(np.isfinite(f.width).all() and f.width.gt(0).all()),current_head_policy_passed=diagnostic['current_head_policy_passed'],routing_ready=False)
    f['routing_ready']=False
    f.to_parquet(folder/'trial_fragment_attributes.parquet',index=False)
    qa=dict(widths_finite_positive=bool(np.isfinite(f.width).all() and f.width.gt(0).all()),positive_slopes=bool(f.slope.gt(0).all()),
        head_reconstruction=heads_qa,explicit_fixed_nodes=sorted(map(int,fixed)),
        routing_ready=False,remaining_requirements=['Reapply routing width QC to newly measured interior fragments','Regenerate calibrated Cunge K/X and RAPID preparation','Resolve any failed boundary or exterior checks'],
        depth_reference='Saved BASED width-depth relation; no roughness calibration changed')
    (folder/'fragment_attributes_audit.json').write_text(json.dumps(qa,indent=2))
    return dict(example=row.example,state=row.state,split_fragments=int(geometry.source_link.isin(split).sum()),widths_valid=qa['widths_finite_positive'],positive_slopes=qa['positive_slopes'],status='derived',routing_ready=False)


def task(args):
    data,folder,resume=args;folder=Path(folder);output=folder/'fragment_result.json'
    if resume and output.exists():
        cached=json.loads(output.read_text())
        if cached['status']!='blocked' or (folder/'strict_head_failure.json').exists():return cached
    row=pd.Series(data)
    try:result=derive(row,folder)
    except Exception as error:result=dict(example=row.example,state=row.state,status='blocked',reason=str(error),routing_ready=False)
    output.write_text(json.dumps(result,indent=2))
    return result


def main():
    from concurrent.futures import ProcessPoolExecutor
    parser=argparse.ArgumentParser();parser.add_argument('--scope',default='reviewed');parser.add_argument('--workers',type=int,default=1);parser.add_argument('--resume',action='store_true');args=parser.parse_args();out=OUT/args.scope
    registry=pd.read_csv(ROOT/'RAPID/outputs/anchoring_audit_20261008/review_registry.csv',dtype={'example':str}).set_index(['example','state'])
    tasks=[]
    for path in sorted(out.glob('*/*/trial_links.parquet')):
        ex,state=path.parent.parent.name,path.parent.name;row=registry.loc[(ex,state)].copy();row['example']=ex;row['state']=state
        tasks.append((row.to_dict(),str(path.parent),args.resume))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(task,tasks)):
            rows.append(result)
            if (i+1)%25==0 or args.scope=='reviewed':print(i+1,len(tasks),result['example'],result['state'],result['status'],result.get('reason',''),flush=True)
    pd.DataFrame(rows).to_csv(out/'fragment_attribute_results.csv',index=False)
if __name__=='__main__':main()
