"""Explicit geometry/slope-only scope for the broader descriptor audit."""
from pathlib import Path
import hashlib,json
import pandas as pd
import broad_descriptor_audit as broad
OUT=broad.OUT/'geometry_with_manning'
EXCLUDED={'resistance_explicit_depth','delta_G_inverse_depth'}

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ['provenance.json','control_correlations_provenance.json']:
        p=broad.OUT/name;meta=json.loads(p.read_text())
        for src,h in meta['source_sha256'].items():
            assert hashlib.sha256(Path(src).read_bytes()).hexdigest()==h,src
            sources[src]=h
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    catalog=pd.read_csv(broad.OUT/'candidate_catalog.csv');removed=catalog[catalog.variable.isin(EXCLUDED)]
    assert set(removed.variable)==EXCLUDED
    catalog=catalog[~catalog.variable.isin(EXCLUDED)].copy()
    catalog.loc[catalog.variable.eq('parent_slope_slow'),'family']='slope/head'
    catalog.loc[catalog.variable.eq('parent_slope_slow'),'kind']='geometry/slope'
    catalog.loc[catalog.variable.eq('delta_G_manning'),['family','kind','role']]=['Manning-derived geometry','geometry/slope','candidate']
    gm_metadata={'scope':'whole network: child minus parent', 'source':'full_grid_review.geometry_diagnostic; width/connectivity allocation', 'definition':'sum L B^(2/5) f^(3/5) S^(-3/10), child minus parent', 'unit':'m^(7/5) (slope dimensionless)'}
    for col,value in gm_metadata.items():catalog.loc[catalog.variable.eq('delta_G_manning'),col]=value
    allowed=set(catalog.variable)
    assert not catalog.kind.eq('hydraulic reference').any()
    catalog.to_csv(OUT/'candidate_catalog.csv',index=False)
    removed.assign(exclusion_reason='User scope: width, length, slope and topology only; no resistance or inverse-depth G; Manning geometry permitted').to_csv(OUT/'excluded_candidates.csv',index=False)
    names=['coverage.csv','factorial_signatures.csv','response_associations.csv','exploratory_summary.csv','control_correlations.csv','affine_aliases.csv','deferred_inventory.csv']
    for name in names:
        p=broad.OUT/name;frame=pd.read_csv(p)
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        if 'variable' in frame:frame=frame[frame.variable.isin(allowed)].copy()
        if name=='affine_aliases.csv':frame=frame[frame.affine_alias_of.isin(allowed)]
        if 'family' in frame:frame.loc[frame.variable.eq('parent_slope_slow'),'family']='slope/head'
        if 'variable' in frame:
            mask=frame.variable.eq('delta_G_manning')
            for col,value in [('family','Manning-derived geometry'),('kind','geometry/slope'),('role','candidate')]:
                if col in frame:frame.loc[mask,col]=value
            for col,value in gm_metadata.items():
                if col in frame:frame.loc[mask,col]=value
        frame.to_csv(OUT/name,index=False)
    features=pd.read_parquet(broad.OUT/'features.parquet').drop(columns=list(EXCLUDED))
    assert allowed<=set(features)
    features.to_parquet(OUT/'features.parquet',index=False)
    sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    p=broad.OUT/'candidate_catalog.csv';sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    meta=dict(reference='SYN-GEOMETRY-ONLY-DESCRIPTORS-2026-10-30',n_candidates=len(catalog),n_rows=len(features),excluded=sorted(EXCLUDED),source_sha256=sources,scope='Width, length, geometric slope/head and topology/allocation only. Parent inverse-square-root slope retained as slope-only descriptor. No fitted depths, n, Q, resistance or inverse-depth G candidates. Whole-network Manning geometry G_M is included with width/connectivity-derived allocation; interpretation conditional on Manning hydraulics.',fits=0,routing_runs=0)
    for name in ['provenance.json','control_correlations_provenance.json']:(OUT/name).write_text(json.dumps(meta,indent=2))
    print('Geometry-only scope:',len(catalog),'candidates;',len(features),'rows')
if __name__=='__main__':build()
