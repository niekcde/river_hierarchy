"""Broad existing-descriptor mechanism audit. Exploratory; no feature selection or fits."""
from pathlib import Path
from functools import lru_cache
import itertools,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
import control_predictor_screen as cp
import geometry_predictor_screen as legacy
from build import METRICS,HYDRAULIC
OUT=cp.OUTPUT/'analysis/broad_descriptors_20261030'
BASE_METRICS={k:v for k,v in METRICS.items() if k not in HYDRAULIC}
WIDTH={'width_mean','width_sd','width_harmonic','width_p10','width_p90'}
LENGTH={'total_length','route_length_mean','route_length_sd','route_length_min','route_length_max','branch_length_contrast_mean','branch_length_contrast_sd'}
LEGACY_L={'equivalent_length'};LEGACY_W={'equivalent_width'}
EXTRA_SLOPE=['slope_mean','slope_min','slope_max','slope_cv','head_drop']
ROLES=['parent','child','delta']

def scale_metric(name,L,W):
    if name in WIDTH:return W
    if name in LENGTH:return L
    return {'area_proxy':L*W,'bifurcation_density':1/L,'branch_length_variance_sum':L*L,'area_elongation':L/W}.get(name,1.)

@lru_cache(None)
def extended_reference(key):
    pars=dict(zip(cp.SHAPE,key),total_width_m=500.,total_length_m=20000.,reference_slope=1e-4)
    graphs,pairs=cp.make_states(pars);out={}
    for state,g in graphs.items():
        if state.startswith('single'):continue
        if not any(g.nodes[n]['x']==5000 for n in g) or not any(g.nodes[n]['x']==15000 for n in g):continue
        dm=legacy.domain_metrics(g,5000,15000)
        source=next(n for n in g if g.nodes[n]['x']==5000);sink=next(n for n in g if g.nodes[n]['x']==15000)
        paths=list(nx.all_simple_edge_paths(g,source,sink))
        lengths=[sum(g.edges[e]['length_m'] for e in p) for p in paths]
        dm['signed_length_width_association']=dm['equivalent_length']/np.mean(lengths)-1
        edges=[d for u,v,k,d in g.edges(keys=True,data=True) if g.nodes[u]['x']>=5000 and g.nodes[v]['x']<=15000]
        ls=np.array([d['length_m'] for d in edges]);slopes=np.array([d['slope'] for d in edges]);mean=np.average(slopes,weights=ls)
        dm.update(slope_mean=mean,slope_min=slopes.min(),slope_max=slopes.max(),slope_cv=np.sqrt(np.average((slopes-mean)**2,weights=ls))/mean,head_drop=g.nodes[source]['head_m']-g.nodes[sink]['head_m'])
        out[state]=dm
    return out


def catalog_from_reference(r,extra):
    records=[]
    for metric,spec in BASE_METRICS.items():
        for role in [*ROLES,'ratio']:
            name=f'B_{role}_{metric}'
            if name not in r:continue
            records.append(dict(variable=name,metric=metric,role=role,family=spec[2],source='current DAG descriptor library',definition=spec[3],scope='closed outer B domain',kind='geometry/allocation',unit='dimensionless' if role=='ratio' or spec[0]==0 else 'geometry dimensional',status='computed'))
    for metric in extra:
        family='slope/head' if metric in EXTRA_SLOPE else 'shape/path-width' if metric in ['equivalent_length','equivalent_width','elongation','signed_length_width_association'] else 'historical path geometry'
        definition={'equivalent_length':'Path-harmonic-width weighted complete path length','equivalent_width':'Sum of complete-path harmonic widths; not discharge allocated width','elongation':'equivalent_length / equivalent_width','signed_length_width_association':'equivalent_length / unweighted mean path length - 1'}.get(metric,metric+'; see geometry_predictor_screen.domain_metrics or extended_reference')
        for role in ROLES:
            records.append(dict(variable=f'path_{role}_{metric}',metric=metric,role=role,family=family,source='existing path metric definitions' if metric not in EXTRA_SLOPE else 'reference geometric slope/head summaries',definition=definition,scope='closed outer B domain; complete paths for path metrics',kind='geometry/slope',unit='see definition',status='computed'))
    return records


def features():
    manifest,_,_=cp.load_analysis(cp.OUTPUT)
    existing=pd.read_parquet(cp.OUT/'features.parquet');rows=[];catalog=None
    jobs=[j for j in manifest['jobs'] if j['settings']['method']=='fixed_manning']
    for i,j in enumerate(jobs,1):
        p=j['parameters'];key=tuple(p[k] for k in cp.SHAPE);ref=cp.reference_features(key,'manning');ext=extended_reference(key)
        L=p['total_length_m']/20000;W=p['total_width_m']/500;S=p['reference_slope']/1e-4
        for state,r in ref.items():
            parent=state.split('/')[0]+'/original';eparent,echild=ext[parent],ext[state]
            if catalog is None:catalog=catalog_from_reference(r,eparent)
            row=dict(case=j['case'],child_state=state)
            for rec in catalog:
                name,metric,role=rec['variable'],rec['metric'],rec['role']
                if name.startswith('B_'):
                    row[name]=r[name]*(1. if role=='ratio' else scale_metric(metric,L,W))
                else:
                    scale=L if metric in LEGACY_L else W if metric in LEGACY_W else L/W if metric=='elongation' else 1/L if metric=='branching_density_by_length' else S if metric in ['slope_mean','slope_min','slope_max'] else L*S if metric=='head_drop' else 1.
                    row[name]=(eparent[metric] if role=='parent' else echild[metric] if role=='child' else echild[metric]-eparent[metric])*scale
            rows.append(row)
        if i%2000==0:print('Broad geometry',i,flush=True)
    reconstructed=pd.DataFrame(rows)
    overlaps=(set(existing.columns)&set(reconstructed.columns))-{'case','child_state'}
    parity=existing.merge(reconstructed[['case','child_state',*sorted(overlaps)]],on=['case','child_state'],suffixes=('_saved','_new'),validate='many_to_one')
    for col in overlaps:np.testing.assert_allclose(parity[col+'_saved'],parity[col+'_new'],rtol=1e-9,atol=1e-9,equal_nan=True)
    result=existing.merge(reconstructed.drop(columns=list(overlaps)),on=['case','child_state'],validate='many_to_one')
    for c in cp.COLUMNS:
        if c not in [r['variable'] for r in catalog]:
            catalog.append(dict(variable=c,metric=c,role='current predictor' if c!='delta_G_manning' else 'analytical reference',family='current hydraulic' if c in ['resistance_explicit_depth','delta_G_inverse_depth','delta_G_manning','parent_slope_slow'] else 'current transformed/scale',source='current predictor bridge',definition=cp.LABELS[c],scope='current B-domain/source-unit conventions',kind='hydraulic reference' if c in ['resistance_explicit_depth','delta_G_inverse_depth','delta_G_manning'] else 'geometry/slope',unit='see current definitions',status='computed'))
    from spread_entropy_comparison import OUT as SDOUT,COLUMNS as SD_COLUMNS
    extra=pd.read_parquet(SDOUT/'features.parquet')[['case','method','child_state',*SD_COLUMNS]]
    result=result.merge(extra,on=['case','method','child_state'],validate='one_to_one')
    for c in SD_COLUMNS:catalog.append(dict(variable=c,metric=c,role='candidate difference',family='spread/entropy differences',source='spread_entropy_comparison',definition=c,scope='closed outer B domain',kind='geometry/allocation',unit='see definition',status='computed'))
    catalog=pd.DataFrame(catalog);assert not catalog.variable.duplicated().any()
    # Internal consistency: scale reconstruction and current predictor semantics.
    np.testing.assert_allclose(result.B_delta_route_length_mean/1000,result.length_delta_km,atol=1e-9)
    np.testing.assert_allclose(result.B_delta_route_entropy,result.route_entropy_change,atol=1e-10)
    np.testing.assert_allclose(result.B_delta_route_length_sd,result.route_spread_change_m,atol=1e-8)
    return manifest,result,catalog


def audit(f,catalog,grid):
    columns=catalog.variable.tolist();index=pd.MultiIndex.from_product(grid.values(),names=list(grid));shape=tuple(map(len,grid.values()))
    effects=[];status=[];associations=[];signatures={}
    for (method,state),g in f.groupby(['method','child_state']):
        g=g.set_index(list(grid)).reindex(index)
        ys={y:g[y].to_numpy() for y in cp.RESPONSES}
        for y,v in ys.items():
            s=cp.tensor_stats(v.reshape(shape));signatures[method,state,y]=s
        for col in columns:
            a=g[col].to_numpy();valid=np.isfinite(a);finite=a[valid];tol=1e-10*max(float(np.max(abs(finite))) if len(finite) else 0.,1.)
            varying=bool(len(finite)>1 and np.ptp(finite)>tol)
            state_label='undefined' if not len(finite) else 'partial' if not valid.all() else 'varying' if varying else 'constant_within_type'
            status.append(dict(method=method,child_state=state,variable=col,status=state_label,n=len(a),finite=int(valid.sum()),finite_pct=100*valid.mean(),minimum=finite.min() if len(finite) else np.nan,maximum=finite.max() if len(finite) else np.nan))
            stats=cp.tensor_stats(a.reshape(shape)) if valid.all() else None
            available=stats is not None and not stats['constant']
            if available:
                for i,t in enumerate(grid):effects.append(dict(method=method,child_state=state,variable=col,term=t,order=1,variance_pct=100*stats['main'][i],total_pct=100*stats['total'][i]))
                for k,t in enumerate(itertools.combinations(grid,2)):effects.append(dict(method=method,child_state=state,variable=col,term=' × '.join(t),order=2,variance_pct=100*stats['pairs'][k],total_pct=np.nan))
            for y,v in ys.items():
                response_resolved=np.ptp(v)>(4/60 if y.startswith('centroid') else .5)
                rho=pd.Series(np.round(finite/tol)*tol).corr(pd.Series(v[valid]),method='spearman') if varying and response_resolved else np.nan
                # Squared Hellinger-style difference over disjoint variance components incl higher orders.
                distance=np.nan
                if available and response_resolved:
                    ysx=signatures[method,state,y]
                    def vector(s):
                        z=np.r_[s['main'],s['pairs']];return np.r_[z,max(0.,1-z.sum())]
                    aa,bb=vector(stats),vector(ysx);distance=float(.5*np.sum((np.sqrt(np.maximum(aa,0))-np.sqrt(np.maximum(bb,0)))**2))
                associations.append(dict(method=method,child_state=state,variable=col,response=y,spearman=rho,finite_pct=100*valid.mean(),response_resolved=response_resolved,variance_signature_distance=distance))
        print('Audited',method,state,flush=True)
    return pd.DataFrame(status),pd.DataFrame(effects),pd.DataFrame(associations)


def main():
    OUT.mkdir(parents=True,exist_ok=True);manifest,f,catalog=features()
    catalog.to_csv(OUT/'candidate_catalog.csv',index=False);f.to_parquet(OUT/'features.parquet',index=False)
    status,effects,assoc=audit(f,catalog,manifest['config']['grid'])
    status.to_csv(OUT/'coverage.csv',index=False);effects.to_csv(OUT/'factorial_signatures.csv',index=False);assoc.to_csv(OUT/'response_associations.csv',index=False)
    summary=assoc[assoc.response_resolved].groupby(['method','response','variable']).agg(mean_abs_spearman=('spearman',lambda x:x.abs().mean()),mean_signature_distance=('variance_signature_distance','mean'),types_with_correlation=('spearman','count'),types_with_complete_signature=('variance_signature_distance','count'),min_finite_pct=('finite_pct','min')).reset_index().merge(catalog,on='variable',validate='many_to_one')
    summary.to_csv(OUT/'exploratory_summary.csv',index=False)
    # Exact affine aliases on the full variable-Manning population; not a model-selected pruning.
    v=f[f.method.eq('variable_manning')];groups={};duplicates=[]
    for col in catalog.variable:
        a=v[col].to_numpy();valid=np.isfinite(a)
        if not valid.all() or np.ptp(a)<=1e-10*max(np.max(abs(a)),1):continue
        a=(a-a.mean())/a.std();first=a[np.flatnonzero(abs(a)>1e-8)[0]];a*=np.sign(first)
        key=hashlib.sha256(np.round(a,8).tobytes()).hexdigest()
        if key in groups:duplicates.append(dict(variable=col,affine_alias_of=groups[key],scope='full variable-Manning grid; rounded standardized values to 8 decimals'))
        else:groups[key]=col
    pd.DataFrame(duplicates,columns=['variable','affine_alias_of','scope']).to_csv(OUT/'affine_aliases.csv',index=False)
    exclusions=[dict(item='Routed travel times, celerity, diffusivity and dispersion descriptors',reason='Deferred: response-proximal hydraulic diagnostics; not silently mixed with geometric candidates.'),dict(item='Historical omega, exposure and length fraction',reason='Deferred: earlier matched-unit scope differs from current closed B-domain convention; needs explicit domain reconciliation.'),dict(item='Pixel-width extremes, sinuosity from map geometry and other unavailable observations',reason='Not represented by synthetic scalar-width/length graphs.')]
    pd.DataFrame(exclusions).to_csv(OUT/'deferred_inventory.csv',index=False)
    sources=[Path(__file__),Path(cp.__file__),Path(cp.bridge.__file__),Path(legacy.__file__),cp.ROOT/'RAPID/investigations/parent_child_predictor_exploration/features.py',cp.ROOT/'RAPID/investigations/hydrograph_predictors_20261014/build.py',cp.ROOT/'RAPID/src/rapid_tools/current_domains.py',cp.ROOT/'hierarchy_level_definition/metrics/unit_metrics.py',cp.OUT/'features.parquet',cp.OUT/'provenance.json']
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='SYN-BROAD-DESCRIPTORS-2026-10-30',n_candidates=len(catalog),n_rows=len(f),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Current DAG descriptors and existing historical complete-path metrics on common outer B domains; legacy path-width weights differ from flow allocations and are labelled separately.',fits=0,routing_runs=0,selection='No final subset selected; full-data descriptive screen, not held-out prediction.'),indent=2))
    print('COMPLETE',len(catalog),'candidates;',len(f),'comparisons;',len(duplicates),'affine aliases',flush=True)

if __name__=='__main__':main()
