"""Control -> predictor mechanism audit; no fitting and no routing.

Current real-model ten features, plus delta_G_manning. Depth features use an
explicit reference snapshot (BASED peak or Manning midpoint, also for MCT).
"""
from pathlib import Path
import sys, json, hashlib, itertools
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import current_predictor_bridge as bridge
from synthetic_experiments.full_grid_review import load_analysis, FACTOR_LABELS, TREATMENTS, RESPONSES
from synthetic_experiments.networks import make_states, detect_units
from rapid_tools.reference_section import predict_based_depth
ROOT=bridge.ROOT
OUTPUT=Path('/Volumes/SyntheticResults20261027')
OUT=OUTPUT/'analysis/control_predictors_20261029'
SETTINGS=bridge.SETTINGS
COLUMNS=json.loads(SETTINGS.read_text())['depth_informed_10']['columns']+['delta_G_manning']
LABELS=dict(zip(COLUMNS,['Route mean ratio','Selected domain size','Route spread ratio','Depth resistance change','Width CV ratio','Route entropy ratio','Parent width dominance','Parent slope factor','Route length change','Inverse-depth G change','Manning G change']))
SHAPE=['outer_a_fraction','inner_fraction','length_multiplier','internal_location','branch_width_factor']

@lru_cache(None)
def reference_features(values,kind):
    pars=dict(zip(SHAPE,values),total_width_m=500.,total_length_m=20000.,reference_slope=1e-4)
    graphs,pairs=make_states(pars);tables={};edgeids={}
    for state,g in graphs.items():
        nodes={n:i for i,n in enumerate(g)};rows=[];lookup={}
        for i,(u,v,k,d) in enumerate(g.edges(keys=True,data=True),1):
            lookup[u,v,k]=i
            rows.append(dict(id_link=i,id_us_node=nodes[u],id_ds_node=nodes[v],link_length_m=d['length_m'],width=d['w'],slope=d['slope'],x0=g.nodes[u]['x'],x1=g.nodes[v]['x']))
        f=pd.DataFrame(rows)
        alloc=bridge.allocation_tables(f)[1].set_index('id_link')
        # Allocation table's edge flow fractions are mean length contributions / physical length.
        fraction=alloc.loc[f.id_link,'mean_length_contribution_m'].to_numpy()/f.link_length_m.to_numpy()
        f['depth_m']=(f.width/500.)**.65 if kind=='based' else (fraction/(f.width/500.*np.sqrt(f.slope/1e-4)))**.6
        tables[state]=f;edgeids[state]=lookup
    result={}
    for pair in pairs:
        state=pair['parent_state'];g=graphs[state];units=detect_units(g);t=pair['child_state'].split('/')[1]
        selected=set().union(*(e for _,e in units))
        if t=='inner_only':selected=min((e for _,e in units),key=len)
        elif t in ['upstream_only','downstream_only']:
            selected=sorted((e for _,e in units),key=lambda es:min(g.nodes[u]['x'] for u,v,k in es))[0 if t=='upstream_only' else 1]
        fs=[]
        for key in [state,pair['child_state']]:
            full=tables[key];part=full[full.x0.ge(5000)&full.x1.le(15000)].copy()
            part,_=bridge.validate_domain(full,part);fs.append(part)
        row=bridge.features(*fs,{edgeids[state][e] for e in selected})
        if abs(row['delta_G_inverse_depth']) <= 1e-10*max(abs(row['parent_G_inverse_depth']),abs(row['child_G_inverse_depth']),1.):
            row['delta_G_inverse_depth']=0.
        if abs(row['resistance_explicit_depth'])<1e-10:row['resistance_explicit_depth']=0.
        result[pair['child_state']]=row
    return result

@lru_cache(None)
def based_reference(width,slope):
    return predict_based_depth(discharge_cms=1000.,width_m=width,slope=slope)


def feature_table(manifest,responses,analysis):
    rows=[]
    for i,j in enumerate((j for j in manifest['jobs'] if j['settings']['method']=='fixed_manning'),1):
        p=j['parameters'];key=tuple(p[k] for k in SHAPE)
        for method,kind in [('fixed_based','based'),('fixed_manning','manning'),('variable_manning','manning')]:
            h=based_reference(p['total_width_m'],p['reference_slope']) if kind=='based' else (.025*750/(p['total_width_m']*np.sqrt(p['reference_slope'])))**.6
            for state,r in reference_features(key,kind).items():
                f={c:r[c] for c in COLUMNS if c!='delta_G_manning'}
                f['domain_size']*=p['total_length_m']/20000
                f['length_delta_km']*=p['total_length_m']/20000
                f['parent_slope_slow']*=np.sqrt(1e-4/p['reference_slope'])
                f['delta_G_inverse_depth']*=p['total_length_m']/20000*np.sqrt(1e-4/p['reference_slope'])/h
                rows.append(dict(case=j['case'],method=method,child_state=state,**f))
        if i%1000==0:print('Features',i,flush=True)
    f=pd.DataFrame(rows)
    geom=pd.read_parquet(analysis/'geometry_diagnostic.parquet')[['case','method','child_state','delta_G_manning']]
    return responses.merge(f,on=['case','method','child_state'],validate='one_to_one').merge(geom,on=['case','method','child_state'],validate='one_to_one')


def tensor_stats(array):
    """Complete factorial, uniform weights. Missing values never imputed."""
    if not np.isfinite(array).all():return None
    mu=array.mean();var=np.mean((array-mu)**2)
    tol=1e-10*max(float(np.max(np.abs(array))),1.)
    if np.ptp(array)<=tol:return dict(main=np.zeros(array.ndim),total=np.zeros(array.ndim),pairs=np.zeros(array.ndim*(array.ndim-1)//2),constant=True)
    main=[];total=[];means=[]
    for i in range(array.ndim):
        m=array.mean(axis=tuple(j for j in range(array.ndim) if j!=i));means.append(m)
        main.append(np.mean((m-mu)**2)/var)
        total.append(1-np.var(array.mean(axis=i))/var)
    pairs=[]
    for i,j in itertools.combinations(range(array.ndim),2):
        m=array.mean(axis=tuple(k for k in range(array.ndim) if k not in [i,j]))
        pairs.append(np.mean((m-means[i][:,None]-means[j][None,:]+mu)**2)/var)
    return dict(main=np.array(main),total=np.array(total),pairs=np.array(pairs),constant=False)


def matched_stats(array,axis):
    a=np.moveaxis(array,axis,-1).reshape(-1,array.shape[axis]);valid=np.isfinite(a).all(axis=1)
    a=a[valid];finite=array[np.isfinite(array)]
    tol=1e-10*max(float(np.max(np.abs(finite))) if len(finite) else 0.,1.)
    varying=np.ptp(a,axis=1)>tol
    rho=np.full(len(a),np.nan)
    if varying.any():
        v=a[varying].copy()
        for j in range(v.shape[1]):
            for k in range(j+1,v.shape[1]):
                close=np.abs(v[:,k]-v[:,j])<=tol
                v[close,k]=v[close,j]
        ranks=rankdata(v,axis=1);ranks-=ranks.mean(axis=1,keepdims=True)
        x=np.arange(a.shape[1],dtype=float);x-=x.mean()
        ids=np.flatnonzero(varying);den=np.sqrt(np.sum(ranks*ranks,axis=1)*np.sum(x*x));nonconstant=den>0
        varying[ids[~nonconstant]]=False
        rho[ids[nonconstant]]=(ranks[nonconstant]@x)/den[nonconstant]
    sd=np.std(finite) if len(finite) else np.nan
    delta=a[:,-1]-a[:,0]
    return dict(n_contexts=len(valid),valid_contexts=int(valid.sum()),varying_contexts=int(varying.sum()),constant_contexts=int((~varying).sum()),rho_median=float(np.nanmedian(rho)) if varying.any() else np.nan,rho_p10=float(np.nanquantile(rho,.1)) if varying.any() else np.nan,rho_p90=float(np.nanquantile(rho,.9)) if varying.any() else np.nan,positive_context_pct=100*np.mean(delta>tol) if len(a) else np.nan,negative_context_pct=100*np.mean(delta < -tol) if len(a) else np.nan,median_delta=float(np.median(delta)) if len(a) else np.nan,median_abs_delta_sd=float(np.median(abs(delta))/sd) if len(a) and sd>tol else 0. if len(a) else np.nan)


def analyze(f,grid):
    factors=list(grid);index=pd.MultiIndex.from_product(grid.values(),names=factors);shape=tuple(map(len,grid.values()))
    controls=[];effects=[];coverage=[]
    for (method,state),g in f.groupby(['method','child_state']):
        g=g.set_index(factors).reindex(index)
        for variable in [*COLUMNS,*RESPONSES]:
            a=g[variable].to_numpy().reshape(shape);stats=tensor_stats(a)
            # Same response resolution screen as the existing analysis.
            resolved=stats is not None and not stats['constant']
            if variable in RESPONSES:resolved &= np.ptp(a)>(.5 if variable.startswith('peak') else 4/60)
            meta=dict(method=method,child_state=state,variable=variable)
            coverage.append(dict(**meta,n=len(g),finite=int(np.isfinite(a).sum()),complete=stats is not None,resolved=resolved))
            for i,factor in enumerate(factors):
                controls.append(dict(**meta,factor=factor,**matched_stats(a,i)))
                effects.append(dict(**meta,term=factor,order=1,variance_pct=100*stats['main'][i] if resolved else np.nan,total_pct=100*stats['total'][i] if resolved else np.nan))
            for k,(i,j) in enumerate(itertools.combinations(range(len(factors)),2)):
                effects.append(dict(**meta,term=factors[i]+' × '+factors[j],order=2,variance_pct=100*stats['pairs'][k] if resolved else np.nan,total_pct=np.nan))
    return pd.DataFrame(controls),pd.DataFrame(effects),pd.DataFrame(coverage)


def heatmap(controls,method,state,value,title):
    g=controls[(controls.method==method)&(controls.child_state==state)&controls.variable.isin(COLUMNS)]
    p=g.pivot(index='factor',columns='variable',values=value).reindex(index=FACTOR_LABELS,columns=COLUMNS)
    fig,ax=plt.subplots(figsize=(13,5),constrained_layout=True);cmap=plt.get_cmap('coolwarm' if value=='rho_median' else 'viridis').copy();cmap.set_bad('#ddd')
    im=ax.imshow(p,aspect='auto',cmap=cmap,vmin=-1 if value=='rho_median' else 0,vmax=1 if value=='rho_median' else None)
    ax.set(xticks=range(len(COLUMNS)),xticklabels=[LABELS[c] for c in COLUMNS],yticks=range(len(p)),yticklabels=[FACTOR_LABELS[x] for x in p.index],title=title+' · '+state)
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right');fig.colorbar(im,ax=ax);return fig


def effect_figure(effects,method,state):
    g=effects[(effects.method==method)&(effects.child_state==state)]
    variables=[*RESPONSES,*COLUMNS];p=g.pivot(index='term',columns='variable',values='variance_pct').reindex(columns=variables)
    ranked=p[list(RESPONSES)].max(axis=1).sort_values(ascending=False).head(12).index;p=p.loc[ranked]
    fig,ax=plt.subplots(figsize=(14,7),constrained_layout=True);cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#ddd');im=ax.imshow(p,aspect='auto',cmap=cmap,vmin=0,vmax=100)
    ax.set(xticks=range(len(variables)),xticklabels=[RESPONSES.get(c,LABELS.get(c,c)) for c in variables],yticks=range(len(p)),yticklabels=[s.replace('_',' ') for s in p.index],title='Do predictors represent the controls driving the responses? · '+state)
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right');ax.axvline(1.5,color='white',lw=3);fig.colorbar(im,ax=ax,label='% of each variable’s own grid variance');return fig


def main():
    manifest,d,analysis=load_analysis(OUTPUT);OUT.mkdir(parents=True,exist_ok=True)
    f=feature_table(manifest,d,analysis);f.to_parquet(OUT/'features.parquet',index=False)
    c,e,coverage=analyze(f,manifest['config']['grid'])
    for name,table in [('matched_controls',c),('factorial_signatures',e),('coverage',coverage)]:table.to_csv(OUT/(name+'.csv'),index=False)
    for state in TREATMENTS:
        fig=effect_figure(e,'variable_manning',state);fig.savefig(OUT/(state.replace('/','_')+'_mechanisms.png'),dpi=150);plt.close(fig)
    provenance=dict(reference='SYN-CONTROL-PREDICTORS-2026-10-29',rows=len(f),predictors=COLUMNS,source_response_sha256=hashlib.sha256((OUTPUT/'responses.csv').read_bytes()).hexdigest(),source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(bridge.__file__),SETTINGS,ROOT/'RAPID/investigations/hydrograph_predictors_20261014/build.py',ROOT/'RAPID/investigations/parent_child_predictor_exploration/features.py',ROOT/'RAPID/src/rapid_tools/current_domains.py']},depth_snapshot='BASED at peak Q=1000; Manning at Q=750, including variable-depth method. Not response-derived.',undefined_policy='Complete-grid ANOVA withheld if any undefined values; matched statistics only on fully finite three-level contexts; coverage reported. No imputation.',scaling='Homogeneous width/length/slope scaling of shared feature implementation, preserving source unit membership in closed outer corridor.',fits=0,routing_runs=0)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2));print('Complete',len(f),flush=True)

if __name__=='__main__':main()
