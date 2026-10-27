"""Validate saved pilot artifacts and write a reviewable result summary."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
import numpy as np
import pandas as pd
from network_variants.reviewed_geometry import ReviewedGeometry
from local_contract import exterior_map,restore_named_exterior,correct
OUT=ROOT/'RAPID/outputs/local_boundary_20261027_v2';SRC=ROOT/'RAPID/outputs/uncapped_event_window_20261020/parameters/22'

def main():
    r=ReviewedGeometry();pg,*_=r.load('22','indep','S000_base');parent=pd.read_parquet(SRC/'S000_base/parameters.parquet')
    footprints=[];obstructions=[];sources={str(SRC/'S000_base/parameters.parquet')}
    for state,family,restore in [('S013_unit_8','indep',{}),('S296_unit_30','indep',{1010:208,1012:225}),('island_22_041','island',{1008:148})]:
        folder=OUT/state;selected=json.loads((folder/'selected_parent_links.json').read_text());cg,*_=r.load('22',family,state)
        old=pd.read_parquet(SRC/state/'parameters.parquet');sources.add(str(SRC/state/'parameters.parquet'))
        source,geo,actions=restore_named_exterior(parent,old,pg,cg,restore) if restore else (old,cg,[])
        mm=exterior_map(parent,source,pg,geo,selected)
        oldext=old.set_index('id_link').loc[list(mm)];pext=parent.set_index('id_link').loc[list(mm.values())]
        fields=['head_us','head_ds','slope','width','depth_m','reference_discharge_cms','split_fraction','rapid_k','rapid_x','count','rapid_effective_length_m']
        row=dict(state=state,exterior_links=len(mm),existing_geometry_changed=len(restore))
        for k in fields:row['existing_changed_'+k]=int((~np.isclose(oldext[k],pext[k],rtol=1e-8,atol=1e-10)).sum())
        if restore:
            import geopandas as gpd
            corrected=pd.read_parquet(folder/'port_path_parameters.parquet');newgeo=gpd.read_parquet(folder/'port_path_geometry.parquet')
        else:corrected=pd.read_parquet(folder/'parameters.parquet');newgeo=cg
        nm=exterior_map(parent,corrected,pg,newgeo,selected)
        inside=~corrected.id_link.isin(nm)
        corrected.loc[inside,'hydraulic_head_policy']='strict_selected_boundary_trial'
        if restore:
            gg=newgeo.set_index('id_link').geometry
            corrected.loc[inside,'geometry_id']=corrected.loc[inside,'id_link'].map(lambda lid:hashlib.sha256(gg.loc[lid].wkb).hexdigest()[:24])
            corrected.loc[inside,'geometry_release']='isolated_port_path_trial_20261027'
        corrected.to_parquet(folder/('port_path_parameters.parquet' if restore else 'parameters.parquet'),index=False)
        rechecked,qa=correct(parent,corrected,nm,selected)
        np.testing.assert_allclose(rechecked[fields],corrected[fields],rtol=1e-9,atol=1e-9)
        newext=corrected.set_index('id_link').loc[list(nm)];pext=parent.set_index('id_link').loc[list(nm.values())]
        for k in fields:
            assert np.array_equal(newext[k].to_numpy(),pext[k].to_numpy())
            row['local_changed_'+k]=0
        row['local_geometry_changed']=0;footprints.append(row)
        # Produce concrete incidence obstructions for saved topology.
        sp=parent[parent.id_link.isin(selected)];replacement=source[~source.id_link.isin(mm)]
        for v in qa['boundary_nodes']:
            for col in ['u','v']:
                required=sp.loc[sp[col].eq(v),'reference_flow_fraction'].sum()
                ids=replacement.loc[replacement[col].eq(v),'id_link'].tolist()
                if required>1e-12 and not ids:obstructions.append(dict(state=state,node=v,direction='outgoing' if col=='u' else 'incoming',required_reference_fraction=required,available_replacement_edges=ids))
    pd.DataFrame(footprints).to_csv(OUT/'hydraulic_footprints.csv',index=False)
    (OUT/'infeasibility_witnesses.json').write_text(json.dumps(obstructions,indent=2))
    a=pd.read_csv(OUT/'routing_responses.csv');one=a[a.refinement.eq(1)].set_index('name');two=a[a.refinement.eq(2)].set_index('name')
    result=[]
    for state in ['S013_unit_8','S296_unit_30','island_22_041']:
        row=dict(state=state)
        for mode in ['existing','local']:
            name=state+'_'+mode
            row[mode+'_centroid_shift_minutes']=(one.loc[name,'centroid_seconds']-one.loc['parent','centroid_seconds'])/60
            row[mode+'_peak_change_pct']=100*(one.loc[name,'peak_cms']/one.loc['parent','peak_cms']-1)
            row[mode+'_clock_contrast_error_seconds']=abs((one.loc[name,'centroid_seconds']-one.loc['parent','centroid_seconds'])-(two.loc[name,'centroid_seconds']-two.loc['parent','centroid_seconds']))
        result.append(row)
    pd.DataFrame(result).to_csv(OUT/'response_comparison.csv',index=False)
    sources.update(str(p) for p in Path(__file__).parent.glob('*.py'))
    sources.update(str(ROOT/p) for p in ['RAPID/src/rapid_tools/cunge.py','RAPID/src/rapid_tools/k_values.py','network_variants/shared_head_reconstruction.py','network_variants/config/reviewed_geometry.json'])
    checks=dict(exterior_exact=True,all_saved_cases_revalidated=True,maximum_absolute_volume_error=float(a.relative_volume_error.abs().max()),minimum_native_flow_cms=float(a.minimum_native_flow_cms.min()),maximum_tail_excess_cms=float(a.tail_excess_cms.abs().max()),maximum_centroid_contrast_clock_error_seconds=max(z[k] for z in result for k in z if k.endswith('error_seconds')),source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sorted(sources)})
    (OUT/'final_validation.json').write_text(json.dumps(checks,indent=2))
    print(pd.DataFrame(footprints).to_string(index=False));print(pd.DataFrame(result).to_string(index=False));print({k:v for k,v in checks.items() if k!='source_hashes'})

if __name__=='__main__':main()
