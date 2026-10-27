"""Prepare all 15 segmented chains with recorded parent-anchor restoration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point,LineString
from geometry_audit import OUT,ROOT,pilot
from edited_features import pair
from rapid_tools import event_window_batch as batch


def restore_endpoints(line,source,sink):
    """Restore known mapped parent anchors; do not infer nearest node identity."""
    coordinates=list(line.coords);offsets=(Point(coordinates[0]).distance(source),Point(coordinates[-1]).distance(sink))
    if max(offsets)<=1e-5:return line,offsets
    coordinates[0]=(source.x,source.y);coordinates[-1]=(sink.x,sink.y)
    # Removing coincident consecutive vertices does not alter the line.
    clean=[coordinates[0]]
    for c in coordinates[1:]:
        if c!=clean[-1]:clean.append(c)
    result=LineString(clean)
    if not result.is_valid or not result.is_simple or result.length<=0:raise ValueError('Anchor restoration produces invalid/self-intersecting section')
    return result,offsets


def reparameterize(row,new_length):
    row=row.copy();r=row.iloc[0];q=float(r.reference_discharge_cms)
    kwargs=dict(n_bounds=(float(r.grid_n_min),float(r.grid_n_max)),alpha=float(r.cunge_alpha),min_celerity=float(r.celerity_min_mps),max_celerity=float(r.celerity_max_mps))
    original=pilot.cunge.freeze_grid(row,q,**kwargs);mult=float(r['count'])/float(original['count'].iloc[0])
    if not np.isclose(mult,round(mult)) or mult<1:raise ValueError('Cannot reproduce prior cell refinement')
    row['link_length_m']=new_length;row['slope']=(row.head_us-row.head_ds)/new_length
    grid=pilot.cunge.freeze_grid(row,q,**kwargs);grid['count']*=int(round(mult));grid['rapid_effective_length_m']=grid.link_length_m/grid['count']
    return pilot.cunge.parameters(grid,float(r.n_manning),beta=float(r.diffusion_multiplier))


def main():
    OUT.mkdir(exist_ok=True);plan=pilot.read(pilot.PLAN);blocks=pilot.read(pilot.OUT/'eligible_blocks.json');audit=pilot.read(OUT/'source_anchor_audit.csv')
    jobs_new=[];repairs=[];features=[];counts=[]
    for case in plan['cases']:
        ex=case['example'];jobs={j['state_id']:j for j in case['jobs']};parent=pilot.read(jobs['S000_base']['parameters']);assert pilot.sha(jobs['S000_base']['parameters'])==case['source_sha256'][jobs['S000_base']['parameters']]
        parent_geo=pilot.geometry_frame(Path(jobs['S000_base']['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg').set_index('id_link');coords={}
        for r in parent.itertuples():
            line=parent_geo.loc[r.id_link].geometry;coords[int(r.u)]=Point(line.coords[0]);coords[int(r.v)]=Point(line.coords[-1])
            np.testing.assert_allclose(line.length,r.link_length_m,rtol=1e-9,atol=1e-6)
        sections={};geometries={}
        for b in blocks[ex]:
            state=b['state'];job=jobs[state];full=pilot.read(job['parameters']);assert pilot.sha(job['parameters'])==case['source_sha256'][job['parameters']]
            saved=pilot.read(pilot.SCOPES/'graphs'/f'{b["child_graph"]}.parquet');section,_=pilot.validate_domain(full,saved)
            original_geo=pilot.geometry_frame(Path(job['parameters']).parents[3]/'variant/widths/links_with_width_families.gpkg').set_index('id_link').loc[b['child_id']].geometry
            line,offsets=restore_endpoints(original_geo,coords[b['source']],coords[b['sink']]);new=section
            # Use audited identity and offset; refuse deviations beyond the observed small-offset cases.
            recorded=audit[(audit.example==ex)&(audit.state==state)].iloc[0]
            np.testing.assert_allclose(offsets,[recorded.inlet_offset_m,recorded.outlet_offset_m],rtol=1e-8,atol=1e-8)
            if max(offsets)>15:raise ValueError('Endpoint displacement exceeds reviewed 15 m envelope')
            if max(offsets)>1e-5:new=reparameterize(section,line.length)
            sections[state]=new;geometries[state]=line
            repairs.append(dict(example=ex,state=state,restored=max(offsets)>1e-5,inlet_offset_m=offsets[0],outlet_offset_m=offsets[1],old_length_m=float(section.link_length_m.iloc[0]),new_length_m=float(new.link_length_m.iloc[0]),length_change_pct=float(100*(new.link_length_m.iloc[0]/section.link_length_m.iloc[0]-1)),old_slope=float(section.slope.iloc[0]),new_slope=float(new.slope.iloc[0]),old_cells=int(section['count'].iloc[0]),new_cells=int(new['count'].iloc[0])))
        child,lineage=pilot.compose(parent,blocks[ex],sections);g=pilot.graph(child)
        assert all(g.in_degree(n)<=1 and g.out_degree(n)<=1 for n in g)
        state='all_closed_units_collapsed';dest=OUT/'variants'/ex/state;dest.mkdir(parents=True,exist_ok=True)
        child.to_parquet(dest/'parameters.parquet',index=False);lineage.to_csv(dest/'lineage.csv',index=False)
        lines=[geometries[r.block_state] if r.section_role=='collapsed_block' else parent_geo.loc[r.id_link].geometry for r in child.itertuples()]
        geometry=gpd.GeoDataFrame(child[['id_link','u','v','section_role','block_state','link_length_m','width','depth_m','slope']].copy(),geometry=lines,crs=parent_geo.crs)
        np.testing.assert_allclose(geometry.geometry.length,child.link_length_m,rtol=1e-9,atol=1e-6);geometry.to_file(dest/'links.gpkg',driver='GPKG',index=False)
        parent_ids=set(lineage.loc[lineage.role=='collapsed_block','parent_link_id']);child_ids=set(lineage.loc[lineage.role=='collapsed_block','child_link_id'])
        m=pair(parent,child,parent_ids,child_ids);m.update(example=ex,state=state,kind='segmented',feature_domain='selected closed blocks only; connectors/tails excluded',excluded_parent_length_m=float(parent.loc[~parent.id_link.isin(parent_ids),'link_length_m'].sum()))
        features.append(m);dt=pilot.shared_clock([child],[30,60],min([jobs['S000_base']['native_dt_seconds']]+[jobs[b['state']]['native_dt_seconds'] for b in blocks[ex]]));pilot.equilibrium_check(child,dt)
        jobs_new.append(dict(example=ex,state=state,parameters=str(dest/'parameters.parquet'),native_dt_seconds=dt,aliases=['segmented_v2/'+state]))
        counts.append(dict(example=ex,collapsed_blocks=len(blocks[ex]),unchanged_sections=int(child.section_role.eq('unchanged').sum()),routing_cells=int(child['count'].sum()),native_dt_seconds=dt))
        print(ex,'prepared',len(blocks[ex]),'blocks',flush=True)
    # Same edited-region definition for the existing six pilot pairs.
    oldjobs=pilot.read(pilot.OUT/'prepared_jobs.json');selections=pilot.read(pilot.OUT/'selection.json')
    cases={c['example']:c for c in plan['cases']}
    for selection in selections:
        ex,state=selection['example'],selection['state'];job=next(j for j in oldjobs if j['example']==ex and j['state']==state);base=next(j for j in cases[ex]['jobs'] if j['state_id']=='S000_base');parent=pilot.read(base['parameters']);child=pilot.read(job['parameters']);lineage=pilot.read(Path(job['parameters']).parent/'lineage.csv')
        parent=parent[parent.id_link.isin(selection['parent_ids'])];domain_child_ids=set(lineage.loc[lineage.parent_link_id.isin(selection['parent_ids']),'child_link_id']);child=child[child.id_link.isin(domain_child_ids)]
        edit_parent=set(lineage.loc[lineage.role=='collapsed_block','parent_link_id']);edit_child=set(lineage.loc[lineage.role=='collapsed_block','child_link_id'])
        m=pair(parent,child,edit_parent,edit_child);m.update(example=ex,state=state,kind='pair',feature_domain='selected closed blocks only; connectors excluded',excluded_parent_length_m=float(parent.loc[~parent.id_link.isin(edit_parent),'link_length_m'].sum()));features.append(m)
    pd.DataFrame(features).to_parquet(OUT/'edited_predictors.parquet',index=False);pd.DataFrame(repairs).to_csv(OUT/'anchor_restoration.csv',index=False);pd.DataFrame(counts).to_csv(OUT/'segmentation_summary.csv',index=False);pilot.save(OUT/'prepared_jobs.json',jobs_new)
    for p,h in pilot.SOURCES.items():assert pilot.sha(p)==h
    pilot.save(OUT/'preparation_manifest.json',dict(status='prepared',sources=pilot.SOURCES,code_sha256={str(Path(__file__)):pilot.sha(__file__),str(Path(pilot.__file__)):pilot.sha(pilot.__file__),str(Path(__file__).with_name('edited_features.py')):pilot.sha(Path(__file__).with_name('edited_features.py'))},anchor_policy='Restore only endpoints to explicitly mapped parent boundary coordinates, retain interior geometry and width/depth; recalculate length, head-drop slope, frozen cells, celerity/K/X with unchanged calibrated policies. Max audited offset 15 m. No source modifications.',predictor_policy='Exclude connectors/tails from seven geometric predictors; connected graph retained for probability propagation and zero-reward route moments.'))
    # New plan includes parent and legacy reference reruns, plus pilot pairs/controls where available.
    newcases=[]
    for case in plan['cases']:
        ex=case['example'];c=dict(case);old={j['state_id']:j for j in c['jobs']};receipt=ROOT/f'RAPID/outputs/event_window_runs_20261013/{ex}/completion.json';done=json.loads(receipt.read_text());c['tail_hours']=done['final_tail_hours'];c['simulation_end_utc']=(pd.Timestamp(c['window']['end_utc'])+pd.Timedelta(seconds=30,hours=c['tail_hours'])).isoformat()
        c['jobs']=[old['S000_base'],old['single_centerline']]
        j=next(j for j in jobs_new if j['example']==ex);c['jobs'].append(dict(state_id=j['state'],**{k:v for k,v in j.items() if k not in ['state','example']}))
        for j in oldjobs:
            if j['example']==ex and j['kind'] in ['pair','single_control']:c['jobs'].append(dict(state_id=j['state'],parameters=j['parameters'],native_dt_seconds=j['native_dt_seconds'],aliases=['pilot/'+j['state']]))
        c['source_sha256']={p:pilot.sha(p) for p in [c['forcing'],str(receipt),*[j['parameters'] for j in c['jobs']]]};newcases.append(c)
    routing=dict(schema='reviewed-event-window-batch-v2',status='all_network_segmentation_extension',code_sha256=batch.code_sources(),source_sha256={str(OUT/'preparation_manifest.json'):pilot.sha(OUT/'preparation_manifest.json')},cases=newcases)
    path=OUT/'routing_plan.json'
    if path.exists() and json.loads(path.read_text())!=routing:raise ValueError('Existing plan differs; version outputs before rerouting')
    pilot.save(path,routing)
    print('Ready: 15 segmented states + reference runs + existing pair controls')

if __name__=='__main__':main()
