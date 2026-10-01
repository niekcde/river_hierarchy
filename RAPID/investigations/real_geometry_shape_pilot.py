"""Geometry/slope pilot on a deliberately small real-network subset; no routing.

Authoritative parent domains, geometry-only selection, common event thresholds,
whole-network held-out evaluation. Run with the river-hierarchy-rivgraph Python.
"""
from pathlib import Path
import sys,json,itertools,hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from scipy.integrate import trapezoid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.plotting import (_build_base_context,_load_selected_hydraulic_domain_frame,
                                 _compute_target_base_share,_exposure_fraction)
from scan_event_shape import load
OUT=ROOT/'RAPID/assets/real_geometry_shape_pilot'
IDS=['03','21','55','56']
Y=['lag_shift_pct','temporal_spread_change_pct','peak_excess_change_pct']
X='resistance_predictor';L='parent_equivalent_length';EL='parent_elongation'
D='parent_dominant_width_fraction';CV='parent_path_length_cv'
A='signed_length_width_association';S='domain_reference_slope'
SETS={'resistance_product':[X], 'compact':[X,L,EL,D],
      'compact_plus_arrangement':[X,L,EL,D,CV,A],
      'compact_plus_slope':[X,L,EL,D,S],
      'compact_plus_all':[X,L,EL,D,CV,A,S],
      'separate_components_plus_all':['resistance_log','length_fraction','width_fraction',L,EL,D,CV,A,S]}

def candidates():
    rows=[];issues=[]
    for example in IDS:
      for kind in ['indep','compound_bubbles']:
        exp=ROOT/f'network_variants/output/{example}_{kind}__anchored'
        if not (exp/'rapid_run_registry.csv').exists():continue
        registry=pd.read_csv(exp/'rapid_run_registry.csv')
        trans=_load_selected_hydraulic_domain_frame(exp,registry.state_id.tolist())
        if trans.empty:continue
        ctx=_build_base_context(exp)
        nodes=pd.read_csv(exp/'states/S000_base/hydraulic_context/node_hydraulic_context.csv').set_index('id_node')
        for _,r in trans.iterrows():
          try:
            if r.parent_state_id!='S000_base':raise ValueError('non-base parent')
            hp=exp/'states'/r.state_id/'rapid/run/outlet_hydrograph.csv'
            if not hp.exists():raise ValueError('missing hydrograph')
            ids=[int(x) for x in r.selected_domain_base_link_ids.split(',') if x]
            g=nx.MultiDiGraph()
            for i in ids:
                g.add_edge(int(ctx['us_lookup'][i]),int(ctx['ds_lookup'][i]),key=i,
                           length=float(ctx['len_lookup'][i]),width=float(ctx['width_lookup'][i]))
            u=int(r.parent_upstream_boundary_node);v=int(r.parent_downstream_boundary_node)
            paths=list(itertools.islice(nx.all_simple_edge_paths(g,u,v),10001))
            if not paths or len(paths)>10000:raise ValueError('empty or >10000 paths')
            if set(e for p in paths for e in p)!=set(g.edges(keys=True)):raise ValueError('domain has off-path edges')
            ls=[];ws=[]
            for p in paths:
                lengths=np.array([g.edges[e]['length'] for e in p]);widths=np.array([g.edges[e]['width'] for e in p])
                if np.any(lengths<=0) or np.any(widths<=0):raise ValueError('nonpositive length/width')
                ls.append(lengths.sum());ws.append(lengths.sum()/np.sum(lengths/widths))
            ls=np.array(ls);ws=np.array(ws);leq=np.average(ls,weights=ws)
            row=r.copy();row['matched_base_link_ids']=r.selected_domain_base_link_ids;row['matched_base_upstream_node']=u
            share,_=_compute_target_base_share(row,ctx);exposure,_=_exposure_fraction(row,raw_width_share=share)
            lf=sum(ctx['len_lookup'][i] for i in ids)/ctx['len_lookup'].sum()
            resistance=-np.log10(float(r.Lambda_K))
            dh=float(nodes.loc[u,'rapid_wse_for_slope']-nodes.loc[v,'rapid_wse_for_slope'])
            vals={X:lf*exposure*resistance,L:leq,EL:leq/ws.sum(),D:ws.max()/ws.sum(),
                  CV:ls.std()/ls.mean(),A:leq/ls.mean()-1,S:dh/ls.min()}
            if not all(np.isfinite(list(vals.values()))):raise ValueError('nonfinite predictor')
            if dh<=0:raise ValueError('nonpositive boundary head drop')
            rows.append(dict(example=example,kind=kind,state_id=r.state_id,
                hydraulic_domain_type=r.hydraulic_domain_type,parent_domain_ids=r.selected_domain_base_link_ids,
                upstream_node=u,downstream_node=v,hydrograph=str(hp),
                base_hydrograph=str(exp/'states/S000_base/rapid/run/outlet_hydrograph.csv'),
                n_paths=len(paths),path_length_min=ls.min(),path_length_max=ls.max(),
                domain_head_loss_m=dh,reference_length_m=ls.min(),
                resistance_log=resistance,length_fraction=lf,width_fraction=exposure,**vals))
          except (ValueError,KeyError,nx.NetworkXError) as exc:
            issues.append(dict(example=example,kind=kind,state_id=r.state_id,reason=str(exc)))
        print(example,kind,'geometry complete',flush=True)
    pd.DataFrame(issues,columns=['example','kind','state_id','reason']).to_csv(OUT/'excluded_candidates.csv',index=False)
    return pd.DataFrame(rows)

def select(d):
    # Rank-scaled farthest-point sampling; no response is loaded during selection.
    picks=[]
    for _,g in d.groupby(['example','kind'],sort=True):
        g=g.sort_values('state_id');n=min(len(g),16 if g.kind.iloc[0]=='indep' else 4)
        z=g[[L,D,CV,A,'length_fraction',S]].rank(pct=True).to_numpy()
        chosen=[int(np.argmin(np.sum((z-.5)**2,axis=1)))]
        while len(chosen)<n:
            distance=((z[:,None,:]-z[chosen][None,:,:])**2).sum(axis=2).min(axis=1)
            distance[chosen]=-1;chosen.append(int(np.argmax(distance)))
        picks.extend(g.index[chosen])
    return d.loc[picks].reset_index(drop=True)

def event(t,q,start,ref):
    mask=t>=start;t=t[mask];q=q[mask];z=np.maximum(q-ref,0)
    # Trapezoidal integration of sampled discharge, using a relative clock.
    tt=t-t[0];volume=trapezoid(z,tt)
    if volume<=0:raise ValueError('no excess volume')
    mu=trapezoid(tt*z,tt)/volume
    sd=np.sqrt(trapezoid((tt-mu)**2*z,tt)/volume)
    return dict(sd=sd,peak=q.max()-ref,peak_time=t[np.argmax(q)],volume=volume,
                end_excess_fraction=z[-1]/z.max(),start_excess_fraction=z[0]/z.max())

def responses(selected):
    rows=[];audit=[];sampling=[]
    for r in selected.itertuples():
        bt,bq,bi=load(r.base_hydrograph);ct,cq,ci=load(r.hydrograph)
        change=np.flatnonzero(np.abs(bi-bi[0])>max(np.ptp(bi)*1e-6,1e-8))[0]
        start=bt[max(change-1,0)];inpeak=bt[np.argmax(bi)]
        lag=bt[np.argmax(bq)]-inpeak
        # Shared fixed threshold above both background plateaus, varied below.
        reference=max(bi[0],bi[-1]);excess=bq.max()-reference
        pre_error=abs(bq[max(change-1,0)]-bi[0])/bi[0]
        m=(ct>=bt[0])&(ct<=bt[-1])
        input_error=np.max(np.abs(ci[m]-np.interp(ct[m],bt,bi)))/np.ptp(bi)
        # Sampling intervals can differ: compare input curves within 0.1% range.
        if input_error>.001:raise ValueError(f'input mismatch {r.state_id}: {input_error}')
        if pre_error>.001:raise ValueError('parent not spun up before event')
        if lag<=0:raise ValueError('nonpositive parent lag')
        for fraction in [0.,.05,.10]:
            ref=reference+fraction*excess
            b=event(bt,bq,start,ref);c=event(ct,cq,start,ref)
            if max(b['end_excess_fraction'],c['end_excess_fraction'])>.001:raise ValueError('censored event tail')
            if fraction==0:
                b0,c0=b,c
                b2=event(bt[::2],bq[::2],start,ref);c2=event(ct[::2],cq[::2],start,ref)
                sampling.append(dict(example=r.example,kind=r.kind,state_id=r.state_id,
                    spread_coarsening_difference_pp=100*(c2['sd']/b2['sd']-c['sd']/b['sd']),
                    peak_coarsening_difference_pp=100*(c2['peak']/b2['peak']-c['peak']/b['peak'])))
            rows.append(dict(example=r.example,kind=r.kind,state_id=r.state_id,reference_increment=fraction,
                reference_discharge=ref,lag_shift_pct=100*(c['peak_time']-b['peak_time'])/lag,
                temporal_spread_change_pct=100*(c['sd']/b['sd']-1),
                peak_excess_change_pct=100*(c['peak']/b['peak']-1)))
        audit.append(dict(example=r.example,kind=r.kind,state_id=r.state_id,
            parent_lag_hours=lag/3600,parent_temporal_sd_hours=b0['sd']/3600,
            initial_input=bi[0],final_input=bi[-1],reference_discharge=reference,
            parent_spinup_relative_error=pre_error,input_relative_error=input_error,
            parent_dt_seconds=np.median(np.diff(bt)),child_dt_seconds=np.median(np.diff(ct)),
            parent_end_excess_fraction=b0['end_excess_fraction'],child_end_excess_fraction=c0['end_excess_fraction']))
    pd.DataFrame(sampling).to_csv(OUT/'sampling_sensitivity.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'event_audit.csv',index=False)
    return pd.DataFrame(rows)

def evaluate(d, prefix="", transfer=True):
    predictions=[];scores=[];corr=[]
    sets=dict(SETS)
    # Predeclared simple product axes; no search on this small real subset.
    for name,cols in {'product_X_D':[X,D],'product_X_CV':[X,CV],
                      'product_X_A':[X,A],'product_X_S':[X,S],
                      'product_Lf_elongation_D':['length_fraction',EL,D]}.items():
        d[name]=d[cols].prod(axis=1);sets[name]=[name]
    for name,cols in sets.items():
      for response in Y:
        pred=np.empty(len(d));null=np.empty(len(d))
        for example in IDS:
            train=d.example!=example;test=~train
            model=make_pipeline(StandardScaler(),Ridge(alpha=10.))
            model.fit(d.loc[train,cols],d.loc[train,response])
            pred[test]=model.predict(d.loc[test,cols]);null[test]=d.loc[train,response].mean()
        for example in IDS+['pooled']:
            mask=np.ones(len(d),bool) if example=='pooled' else (d.example==example).to_numpy()
            obs=d.loc[mask,response];p=pred[mask]
            scores.append(dict(model=name,response=response,example=example,n=mask.sum(),
                r2=r2_score(obs,p),mae=mean_absolute_error(obs,p),
                mean_baseline_mae=mean_absolute_error(obs,null[mask])))
        predictions.append(pd.DataFrame(dict(model=name,response=response,example=d.example,
            state_id=d.state_id,kind=d.kind,observed=d[response],predicted=pred)))
    for feature in list(dict.fromkeys(sum(sets.values(),[]))):
      for response in Y:
        for example in IDS+['pooled']:
            g=d if example=='pooled' else d[d.example==example]
            rho=spearmanr(g[feature],g[response]).statistic if g[feature].nunique()>1 else np.nan
            corr.append(dict(feature=feature,response=response,example=example,spearman=rho))
    pd.DataFrame(scores).to_csv(OUT/(prefix+'held_out_network_scores.csv'),index=False)
    pd.concat(predictions).to_csv(OUT/(prefix+'held_out_network_predictions.csv'),index=False)
    pd.DataFrame(corr).to_csv(OUT/(prefix+'correlations.csv'),index=False)
    if not transfer:return
    # Frozen synthetic-model transfer; deliberately no adjustment to real responses.
    synth=pd.read_parquet(ROOT/'synthetic_experiments/outputs/full_grid_island/factor_analysis/geometry_predictors/slope_path_arrangement/features.parquet')
    transfer=[];ranges=[]
    for col in SETS['separate_components_plus_all']:
        lo,hi=synth[col].min(),synth[col].max()
        ranges.append(dict(feature=col,synthetic_min=lo,synthetic_max=hi,real_min=d[col].min(),real_max=d[col].max(),outside_fraction=((d[col]<lo)|(d[col]>hi)).mean()))
    for name in ['resistance_product','compact_plus_all']:
      cols=SETS[name]
      for response in Y:
        model=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20260928)
        model.fit(synth[cols],synth[response]);p=model.predict(d[cols])
        transfer.append(dict(model=name,response=response,r2=r2_score(d[response],p),mae=mean_absolute_error(d[response],p),observed_abs_median=d[response].abs().median(),predicted_abs_median=np.median(abs(p))))
    pd.DataFrame(transfer).to_csv(OUT/'synthetic_transfer_scores.csv',index=False)
    pd.DataFrame(ranges).to_csv(OUT/'feature_ranges.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c=candidates();c.to_csv(OUT/'geometry_candidates.csv',index=False)
    s=select(c);s.to_csv(OUT/'selected_geometry.csv',index=False)
    shape=responses(s);shape.to_csv(OUT/'threshold_sensitivity.csv',index=False)
    d=s.merge(shape[shape.reference_increment==0],on=['example','kind','state_id'],validate='one_to_one')
    d.to_csv(OUT/'analysis.csv',index=False)
    with threadpool_limits(limits=1):
        evaluate(d)
        evaluate(d[d.kind=='indep'].reset_index(drop=True).copy(),prefix='independent_',transfer=False)
        for fraction in [.05,.10]:
            alt=s.merge(shape[shape.reference_increment==fraction],on=['example','kind','state_id'],validate='one_to_one')
            evaluate(alt,prefix=f'threshold_{fraction:.2f}_',transfer=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(examples=IDS,n_candidates=len(c),n_selected=len(s),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection='16 independent and up to 4 compound states per example; farthest-point rank geometry; S000_base parents only',
        slope='parent boundary rapid_wse_for_slope drop / shortest parent-domain path length',
        shape_reference='max(initial input, final input); sensitivity +5% and +10% parent peak excess',
        event='first input change through each complete outlet record; positive excess; trapezoidal moments',
        validation='leave one whole network out; standardized raw features; Ridge alpha=10; no tuning',
        excluded_scope='additional island reruns; non-base transitions; incomplete domains or missing predictors',
        caveat='exploratory four-network pilot; thresholded event shape is not an independent physical baseflow separation'),indent=2))
    print(d.groupby('example')[Y].agg(['count','min','max']).to_string())
if __name__=='__main__':main()
