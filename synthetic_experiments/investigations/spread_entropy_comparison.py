"""Compare route-spread/entropy differences with existing ratios; no model changes."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import pandas as pd
import control_predictor_screen as cp
OUT=cp.OUTPUT/'analysis/spread_entropy_comparison_20261029'
COLUMNS=['route_spread_change_m','route_spread_change_over_parent_mean','route_entropy_change']
LABELS={'log1p__B_ratio_route_length_sd':'Existing spread ratio (log1p)',COLUMNS[0]:'Spread change (m)',COLUMNS[1]:'Spread change / parent mean length','log1p__B_ratio_route_entropy':'Existing entropy ratio (log1p)',COLUMNS[2]:'Entropy change'}
COMPARE=list(LABELS)

def differences(row,length_scale=1.):
    mean=row['B_parent_route_length_mean']
    if not np.isfinite(mean) or mean<=0:raise ValueError('Parent mean route length must be positive')
    delta=row['B_child_route_length_sd']-row['B_parent_route_length_sd']
    return dict(route_spread_change_m=delta*length_scale,route_spread_change_over_parent_mean=delta/mean,
                route_entropy_change=row['B_child_route_entropy']-row['B_parent_route_entropy'])

def build():
    manifest,_,_=cp.load_analysis(cp.OUTPUT);OUT.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(cp.OUT/'features.parquet');rows=[]
    for i,j in enumerate((j for j in manifest['jobs'] if j['settings']['method']=='fixed_manning'),1):
        p=j['parameters'];ref=cp.reference_features(tuple(p[k] for k in cp.SHAPE),'manning')
        for state,r in ref.items():
            rows.append(dict(case=j['case'],child_state=state,**differences(r,p['total_length_m']/20000),
                check_spread=r['log1p__B_ratio_route_length_sd'],check_entropy=r['log1p__B_ratio_route_entropy']))
        if i%2000==0:print('Geometry comparison',i,flush=True)
    joined=old.merge(pd.DataFrame(rows),on=['case','child_state'],validate='many_to_one')
    for check,col in [('check_spread','log1p__B_ratio_route_length_sd'),('check_entropy','log1p__B_ratio_route_entropy')]:
        np.testing.assert_allclose(joined[check],joined[col],atol=1e-9,rtol=1e-9,equal_nan=True)
    joined=joined.drop(columns=['check_spread','check_entropy'])
    assert np.isfinite(joined[COLUMNS]).all().all()
    joined.to_parquet(OUT/'features.parquet',index=False)
    grid=manifest['config']['grid'];index=pd.MultiIndex.from_product(grid.values(),names=list(grid));shape=tuple(map(len,grid.values()))
    effects=[];coverage=[]
    for (method,state),g in joined.groupby(['method','child_state']):
        g=g.set_index(list(grid)).reindex(index)
        for col in COLUMNS:
            a=g[col].to_numpy().reshape(shape);s=cp.tensor_stats(a);resolved=not s['constant']
            meta=dict(method=method,child_state=state,variable=col)
            coverage.append(dict(**meta,n=len(g),finite=int(np.isfinite(a).sum()),complete=True,resolved=resolved,minimum=a.min(),maximum=a.max()))
            for i,f in enumerate(grid):effects.append(dict(**meta,term=f,order=1,variance_pct=100*s['main'][i] if resolved else np.nan,total_pct=100*s['total'][i] if resolved else np.nan))
            for k,pair in enumerate(itertools.combinations(grid,2)):effects.append(dict(**meta,term=' × '.join(pair),order=2,variance_pct=100*s['pairs'][k] if resolved else np.nan,total_pct=np.nan))
    pd.DataFrame(effects).to_csv(OUT/'factorial_signatures.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    sources=[Path(__file__),Path(cp.__file__),Path(cp.bridge.__file__),cp.OUT/'features.parquet',cp.OUT/'provenance.json',cp.ROOT/'RAPID/investigations/hydrograph_predictors_20261014/build.py',cp.ROOT/'RAPID/investigations/parent_child_predictor_exploration/features.py',cp.ROOT/'RAPID/src/rapid_tools/current_domains.py']
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='SYN-SPREAD-ENTROPY-2026-10-29',n_rows=len(joined),new_candidates=COLUMNS,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},definitions='All changes child minus parent; normalized SD difference divided by positive parent mean route length. B-domain definitions unchanged. Geometry-only values shared across methods.',routing_runs=0,model_fits=0),indent=2))
    print('Complete:',len(joined),flush=True)

if __name__=='__main__':build()
