"""Read-only analysis of the completed three-method 8-factor synthetic grid.
No routing, predictor fitting, or changes to the source run.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .hydraulics import METHODS, LABELS

DEFAULT_OUTPUT=Path('/Volumes/SyntheticResults20261027')
RESPONSES={'centroid_shift_minutes':'Centroid shift (min)', 'peak_shift_minutes':'Peak shift (min)'}
FACTOR_LABELS={'total_width_m':'Reference width','reference_slope':'Reference slope','outer_a_fraction':'Outer A fraction','inner_fraction':'Inner/crossover fraction','length_multiplier':'Branch length multiplier','total_length_m':'Network extent','internal_location':'Internal location','branch_width_factor':'Branch-width factor'}
TREATMENTS=['simple/full','nested/inner_only','nested/main_island_only','nested/full','crossover/upstream_only','crossover/downstream_only','crossover/full']

def load(output=DEFAULT_OUTPUT):
    output=Path(output)
    if not output.is_dir():raise FileNotFoundError('Mount the PhD-drive container first: see synthetic_experiments/RUNBOOK.md')
    manifest=json.loads((output/'manifest.json').read_text());done=json.loads((output/'completion.json').read_text())
    if done.get('status')!='complete' or done['source']!=manifest['source']:raise ValueError('Incomplete or inconsistent run')
    d=pd.read_csv(output/'responses.csv');grid=manifest['config']['grid'];factors=list(grid)
    expected=int(np.prod([len(v) for v in grid.values()]))
    if len(d)!=expected*len(METHODS)*len(TREATMENTS) or len(d)!=done['n_comparisons']:raise ValueError('Unexpected response count')
    if set(d.method)!=set(METHODS) or set(d.child_state)!=set(TREATMENTS):raise ValueError('Missing methods/treatments')
    if d.duplicated(['method','child_state',*factors]).any():raise ValueError('Duplicate factorial observations')
    if not np.isfinite(d[list(RESPONSES)+['centroid_lag_change_pct']]).all().all():raise ValueError('Nonfinite response')
    for name,levels in grid.items():
        if set(d[name])!=set(levels):raise ValueError('Unexpected factor levels: '+name)
    counts=d.groupby(['method','child_state']).size()
    if not counts.eq(expected).all():raise ValueError('Unbalanced factorial strata')
    groups=d.groupby('group_id').agg(rows=('child_state','size'),methods=('method','nunique'),cases=('case','nunique'))
    if len(groups)!=done['n_groups'] or not (groups.rows.eq(7)&groups.methods.eq(1)&groups.cases.eq(1)).all():raise ValueError('Invalid group mapping')
    return manifest,d


def factorial_tables(d,grid):
    """Orthogonal functional ANOVA under uniform weights on a complete grid."""
    main=[];pairs=[];parts=[];contrasts=[];factors=list(grid)
    for (method,state),g in d.groupby(['method','child_state'],sort=False):
        for response in RESPONSES:
            y=g[response];mu=y.mean();var=float(np.mean((y-mu)**2));means={}
            tol=.25 if response.startswith('peak') else 2/60
            resolved=bool(np.ptp(y)>2*tol)
            base=dict(method=method,child_state=state,response=response,response_variation_resolved=resolved)
            actual_var=var
            if var<=1e-24:var=1.  # normalized results are masked below for constant responses
            main_sum=pair_sum=0.
            for factor in factors:
                means[factor]=g.groupby(factor,observed=True)[response].mean()
                vm=float(np.mean((means[factor]-mu)**2));main_sum+=vm
                others=[f for f in factors if f!=factor]
                without=g.groupby(others,observed=True)[response].mean()
                total=1-float(np.mean((without-mu)**2))/var
                main.append(dict(**base,factor=factor,main_pct=100*vm/var,total_pct=100*max(0,total)))
                pivot=g.pivot(index=others,columns=factor,values=response)
                delta=pivot[grid[factor][-1]]-pivot[grid[factor][0]]
                contrasts.append(dict(**base,factor=factor,from_level=grid[factor][0],to_level=grid[factor][-1],n_contexts=len(delta),mean_minutes=delta.mean(),median_minutes=delta.median(),p10_minutes=delta.quantile(.1),p90_minutes=delta.quantile(.9)))
            for a,b in itertools.combinations(factors,2):
                marginal=g.groupby([a,b],observed=True)[response].mean()
                effect=np.array([v-means[a].loc[x]-means[b].loc[z]+mu for (x,z),v in marginal.items()])
                vp=float(np.mean(effect**2));pair_sum+=vp
                pairs.append(dict(**base,factor_a=a,factor_b=b,variance_pct=100*vp/var))
            residual=1-(main_sum+pair_sum)/var
            if residual < -1e-9:raise ValueError('Invalid ANOVA partition')
            parts.append(dict(**base,variance_minutes_squared=actual_var,main_pct=100*main_sum/var,pair_pct=100*pair_sum/var,higher_order_pct=100*max(0,residual)))
    frames=[pd.DataFrame(x) for x in [main,pairs,parts,contrasts]]
    for frame,cols in zip(frames,[['main_pct','total_pct'],['variance_pct'],['main_pct','pair_pct','higher_order_pct'],[]]):
        frame.loc[~frame.response_variation_resolved,cols]=np.nan
    return tuple(frames)


def summaries(d):
    records=[]
    for (method,state),g in d.groupby(['method','child_state']):
        for response in RESPONSES:
            y=g[response];tol=.25 if response.startswith('peak') else 2/60
            records.append(dict(method=method,child_state=state,response=response,n=len(g),p05=y.quantile(.05),median=y.median(),p95=y.quantile(.95),minimum=y.min(),maximum=y.max(),earlier_pct=100*y.lt(-tol).mean(),near_zero_pct=100*y.abs().le(tol).mean(),later_pct=100*y.gt(tol).mean(),deadband_minutes=tol))
    return pd.DataFrame(records)


def paired_methods(d):
    keys=['case','child_state'];records=[]
    for left,right in itertools.combinations(METHODS,2):
        a=d[d.method.eq(left)];b=d[d.method.eq(right)]
        p=a.merge(b,on=keys,suffixes=('_left','_right'),validate='one_to_one')
        if len(p)!=len(a) or len(p)!=len(b):raise ValueError('Unmatched hydraulic methods')
        for state,g in p.groupby('child_state'):
            for response in RESPONSES:
                x=g[response+'_left'];y=g[response+'_right'];delta=y-x
                tol=.25 if response.startswith('peak') else 2/60
                sign=lambda v:np.where(v.abs()<=tol,0,np.sign(v))
                records.append(dict(method_left=left,method_right=right,child_state=state,response=response,n=len(g),median_difference_minutes=delta.median(),median_abs_difference_minutes=delta.abs().median(),p95_abs_difference_minutes=delta.abs().quantile(.95),max_abs_difference_minutes=delta.abs().max(),spearman=x.corr(y,method='spearman') if min(np.ptp(x),np.ptp(y))>2*tol else np.nan,sign_agreement_pct=100*np.mean(sign(x)==sign(y))))
    return pd.DataFrame(records)


def geometry_diagnostic(manifest,d,output):
    """Whole-network geometry and fixed-Manning travel-time identity, no fitted coefficients.
    f is routed width-allocation fraction; G=sum L B^.4 f^.6 S^(-.3).
    T=.6 n^.6 Qref^(-.4) G for fixed wide-rectangular Manning hydraulics.
    """
    rows=[]
    jobs=[j for j in manifest['jobs'] if j['settings']['method']=='fixed_manning']
    for i,j in enumerate(jobs,1):
        p=pd.read_csv(Path(output)/'groups'/j['id']/'parameters.csv')
        p['G_manning']=p.link_length_m*p.width**.4*p.reference_flow_fraction**.6*p.slope**(-.3)
        p['expected_route_length_m']=p.link_length_m*p.reference_flow_fraction
        p['frozen_travel_seconds']=p.link_length_m/p.rapid_celerity_mps*p.reference_flow_fraction
        agg=p.groupby('state')[['G_manning','expected_route_length_m','frozen_travel_seconds']].sum()
        s=j['settings'];coef=.6*s['n_manning']**.6*((s['baseflow_cms']+s['peak_cms'])/2)**(-.4)
        np.testing.assert_allclose(coef*agg.G_manning,agg.frozen_travel_seconds,rtol=1e-10,atol=1e-7)
        for state in TREATMENTS:
            parent=state.split('/')[0]+'/original';delta=agg.loc[state]-agg.loc[parent]
            rows.append(dict(case=j['case'],child_state=state,delta_G_manning=delta.G_manning,delta_expected_route_length_m=delta.expected_route_length_m,manning_reference_shift_minutes=coef*delta.G_manning/60))
        if i%1000==0:print(f'Geometry diagnostic {i}/{len(jobs)}',flush=True)
    features=pd.DataFrame(rows)
    return d.merge(features,on=['case','child_state'],validate='many_to_one')


def score_physical_diagnostic(features):
    records=[]
    for (method,state),g in features.groupby(['method','child_state']):
        pred=g.manning_reference_shift_minutes.to_numpy()
        for response in RESPONSES:
            y=g[response].to_numpy();err=pred-y;den=np.sum((y-y.mean())**2)
            tol=.25 if response.startswith('peak') else 2/60
            resolved=bool(np.ptp(y)>2*tol)
            records.append(dict(method=method,child_state=state,response=response,n=len(g),response_variation_resolved=resolved,response_span_seconds=np.ptp(y)*60,r2_no_fit=1-np.sum(err**2)/den if den>0 and resolved else np.nan,mae_minutes=np.mean(abs(err)),max_abs_error_minutes=max(abs(err))))
    return pd.DataFrame(records)


def build(output=DEFAULT_OUTPUT):
    output=Path(output);out=output/'analysis/full_grid_20261028';out.mkdir(parents=True,exist_ok=True)
    manifest,d=load(output);main,pairs,parts,contrasts=factorial_tables(d,manifest['config']['grid'])
    tables=dict(main_total_effects=main,pair_interactions=pairs,variance_partition=parts,signed_contrasts=contrasts,response_summary=summaries(d),method_agreement=paired_methods(d))
    featurepath=out/'geometry_diagnostic.parquet'
    # Rebuild explicitly: never reuse a table without verifying its source.
    features=geometry_diagnostic(manifest,d,output)
    fixed=features[features.method.eq('fixed_manning')]
    if (fixed.centroid_shift_minutes-fixed.manning_reference_shift_minutes).abs().max()>2/60+1e-6:
        raise ValueError('Fixed-Manning centroid identity exceeds saved convergence tolerance')
    features.to_parquet(featurepath,index=False)
    tables['physical_diagnostic_scores']=score_physical_diagnostic(features)
    for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False)
    provenance=dict(reference='SYN-FULL-DEPTH-GRID-2026-10-28',source_output=str(output),source_response_sha256=hashlib.sha256((output/'responses.csv').read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),n_groups=manifest['n_groups'],n_comparisons=len(d),factors=manifest['config']['grid'],responses=RESPONSES,scientific_source=manifest['source'])
    (out/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2))
    return tables


def load_analysis(output=DEFAULT_OUTPUT):
    output=Path(output);manifest,d=load(output);out=output/'analysis/full_grid_20261028'
    p=json.loads((out/'analysis_provenance.json').read_text())
    if p['source_response_sha256']!=hashlib.sha256((output/'responses.csv').read_bytes()).hexdigest() or p['analysis_source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Analysis source changed: rebuild the analysis first')
    return manifest,d,out


def summary_figure(table):
    fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for ax,response in zip(axes,RESPONSES):
        for i,method in enumerate(METHODS):
            g=table[(table.method==method)&(table.response==response)].set_index('child_state').loc[TREATMENTS]
            x=np.arange(len(g))+(i-1)*.23;y=g['median'].to_numpy()
            ax.errorbar(x,y,yerr=np.array([y-g.p05,g.p95-y]),fmt='o',capsize=3,label=LABELS[method])
        ax.axhline(0,c='gray',lw=.8);ax.set(xticks=range(len(TREATMENTS)),xticklabels=[s.replace('/','\n').replace('_',' ') for s in TREATMENTS],ylabel=RESPONSES[response],title='Median and grid P05–P95');ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    axes[0].legend(fontsize=8);return fig


def heatmaps(table,response,value,row,order,title,cmap='viridis',signed=False):
    fig,axes=plt.subplots(1,3,figsize=(19,max(5,len(order)*.34)),constrained_layout=True)
    data=table[table.response.eq(response)];bound=max(float(data[value].abs().max()),1e-12)
    for ax,method in zip(axes,METHODS):
        g=data[data.method.eq(method)].pivot(index=row,columns='child_state',values=value).reindex(index=order,columns=TREATMENTS)
        colors=plt.get_cmap(cmap).copy();colors.set_bad('#dddddd')
        im=ax.imshow(g,aspect='auto',vmin=-bound if signed else 0,vmax=bound,cmap=colors)
        ax.set(xticks=range(7),xticklabels=[s.replace('/','\n').replace('_',' ') for s in TREATMENTS],yticks=range(len(order)),yticklabels=[FACTOR_LABELS.get(x,x) for x in order],title=LABELS[method]);ax.tick_params(axis='both',labelsize=8)
    fig.colorbar(im,ax=axes,label='Minutes: highest level minus lowest level' if signed else '% of response variance',shrink=.7)
    fig.suptitle(title+' · '+RESPONSES[response]);return fig


def agreement_figure(d):
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for col,(left,right) in enumerate(itertools.combinations(METHODS,2)):
        a=d[d.method.eq(left)].set_index(['case','child_state']);b=d[d.method.eq(right)].set_index(['case','child_state']).reindex(a.index)
        for ax,response in zip(axes[:,col],RESPONSES):
            x=a[response];y=b[response];im=ax.hexbin(x,y,gridsize=45,mincnt=1,bins='log',cmap='viridis');lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
            ax.set(xlabel=LABELS[left],ylabel=LABELS[right],title=RESPONSES[response]);fig.colorbar(im,ax=ax,label='Comparison count')
    return fig


def physical_figure(features):
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for col,method in enumerate(METHODS):
        g=features[features.method.eq(method)]
        for ax,response in zip(axes[:,col],RESPONSES):
            x=g.manning_reference_shift_minutes;y=g[response];im=ax.hexbin(x,y,gridsize=45,mincnt=1,bins='log');lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.8)
            ax.set(xlabel='Manning reference travel-time change (min)',ylabel=RESPONSES[response],title=LABELS[method]);fig.colorbar(im,ax=ax,label='Comparison count')
    return fig
