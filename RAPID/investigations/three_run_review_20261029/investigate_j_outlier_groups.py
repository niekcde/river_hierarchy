"""Three-group ΔJ outlier investigation: analytic routing moments + saved waves.
No routing, calibration, or new predictor fitting. No transformed parent waves.
"""
from pathlib import Path
import json
from functools import lru_cache
import numpy as np
import pandas as pd
import networkx as nx
from incremental_storage import ROOT,SOURCE,OUT,digest
from incremental_storage_patterns import moment

DEST=OUT/'group_investigation'
SOURCES={}


def read_saved(example,state,name='parameters.parquet'):
    path=SOURCE/'states'/example/state/name
    sha=digest(path);receipt=json.loads((path.parent/'completion.json').read_text())
    assert sha==receipt['files'][str(path)],str(path)
    SOURCES[str(path)]=sha
    return pd.read_parquet(path)


def edge_metrics(f):
    f=f.copy();f['tau_seconds']=f['count']*f.rapid_k
    f['local_variance_s2']=f['count']*f.rapid_k**2*(1-2*f.rapid_x)
    f['mean_contribution_seconds']=f.reference_flow_fraction*f.tau_seconds
    f['diffusion_contribution_s2']=f.reference_flow_fraction*f.local_variance_s2
    np.testing.assert_allclose(f.tau_seconds,f.link_length_m/f.rapid_celerity_mps,rtol=1e-10)
    physical=f.diffusion_multiplier*f.reference_discharge_cms*f.link_length_m/(f.width*f.slope*f.rapid_celerity_mps**3)
    np.testing.assert_allclose(f.local_variance_s2,physical,rtol=1e-10)
    f['cell_relaxation_hours']=f.rapid_k*(1-f.rapid_x)/3600
    f['head_drop_m']=f.slope*f.link_length_m
    f['depth_width_ratio']=f.depth_m/f.width
    return f


def state_moments(f):
    graph=nx.DiGraph();graph.add_edges_from(zip(f.u,f.v))
    source=[n for n in graph if graph.in_degree(n)==0];sink=[n for n in graph if graph.out_degree(n)==0]
    assert len(source)==len(sink)==1
    outgoing={}
    for r in f.itertuples():outgoing.setdefault(r.u,[]).append(r)
    acc={source[0]:(1.,0.,0.)};max_f_error=0.
    for node in nx.topological_sort(graph):
        w,first,second=acc[node];mu=first/w;m2=second/w
        for r in outgoing.get(node,[]):
            weight=w*r.split_fraction;max_f_error=max(max_f_error,abs(weight-r.reference_flow_fraction))
            old=acc.get(r.v,(0.,0.,0.));tau=r.tau_seconds
            acc[r.v]=(old[0]+weight,old[1]+weight*(mu+tau),old[2]+weight*(m2+2*mu*tau+tau*tau))
    w,first,second=acc[sink[0]];mu=first/w
    assert abs(w-1)<1e-8 and max_f_error<1e-8
    np.testing.assert_allclose(mu,f.mean_contribution_seconds.sum(),rtol=1e-9)
    pathvar=second/w-mu*mu
    assert pathvar>=-1e-6*max(mu*mu,1)
    pathvar=max(pathvar,0.);diffvar=float(f.diffusion_contribution_s2.sum());var=pathvar+diffvar
    return dict(mean_min=mu/60,variance_h2=var/3600**2,path_variance_h2=pathvar/3600**2,
        diffusion_variance_h2=diffvar/3600**2,transfer_sd_hours=np.sqrt(var)/3600,
        diffusion_variance_fraction=diffvar/var if var>0 else np.nan,
        min_slope=f.slope.min(),max_depth_m=f.depth_m.max(),max_cell_relaxation_hours=f.cell_relaxation_hours.max(),
        reaches=len(f),max_diffusion_contribution_share=f.diffusion_contribution_s2.max()/diffvar if diffvar else 0.,
        slope_lt1e5_edge_exposure_sum=float(f.loc[f.slope<1e-5,'reference_flow_fraction'].sum()),
        negative_X_reaches=int((f.rapid_x<0).sum()))


def geometry_key(f):
    return pd.MultiIndex.from_arrays([f.u,f.v,f.link_length_m.round(6),f.width.round(6)])


def edge_changes(parent,child):
    p=parent.copy();c=child.copy();p.index=geometry_key(p);c.index=geometry_key(c)
    assert p.index.is_unique and c.index.is_unique
    shared=p.index.intersection(c.index)
    changed=np.zeros(len(shared),dtype=bool)
    for col in ['slope','reference_flow_fraction','depth_m','n_manning','diffusion_multiplier']:
        changed|=~np.isclose(p.loc[shared,col],c.loc[shared,col],rtol=1e-10,atol=1e-12)
    shared_changed=shared[changed]
    records=[]
    for side,f in [('parent',p),('child',c)]:
        for key,r in f.iterrows():
            status='removed' if side=='parent' else 'new'
            if key in shared:status='shared_changed' if key in shared_changed else 'shared_unchanged'
            records.append(dict(side=side,status=status,geometry_key=str(key),**r.to_dict()))
    edges=pd.DataFrame(records)
    # Signed edge contributions, including redistributed flow on surviving geometry.
    edges['signed_J_seconds']=np.where(edges.side.eq('child'),1.,-1.)*edges.mean_contribution_seconds
    edges['signed_diffusion_h2']=np.where(edges.side.eq('child'),1.,-1.)*edges.diffusion_contribution_s2/3600**2
    summary=dict(shared_reaches=len(shared),shared_changed_reaches=len(shared_changed),removed_reaches=len(p)-len(shared),new_reaches=len(c)-len(shared))
    for status in ['removed','new','shared_changed','shared_unchanged']:
        sub=edges[edges.status.eq(status)]
        summary[status+'_delta_J_min']=sub.signed_J_seconds.sum()/60
        summary[status+'_delta_diffusion_h2']=sub.signed_diffusion_h2.sum()
    return summary,edges


def exact_piecewise_integrals(t,dq,center):
    h=np.diff(t);a=dq[:-1];b=dq[1:];mass=h*(a+b)/2
    first=(t[:-1]-center)*mass+h*h*(a/2+(b-a)/3)
    return mass,first


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    path=OUT/'depth_methods/comparisons.parquet';SOURCES[str(path)]=digest(path)
    all_data=pd.read_parquet(path);all_data=all_data[all_data.method.eq('manning_midpoint')].copy()
    path=ROOT/'RAPID/outputs/uncapped_event_window_20261020/validation_group_inventory_20261023/records.parquet';SOURCES[str(path)]=digest(path)
    labels=pd.read_parquet(path)[['example','state','edit_type','selected_unit_ids','compound_ids','nested']]
    primary=all_data[all_data.window.eq('0.5%')].merge(labels,on=['example','state'],how='left',validate='one_to_one')
    primary.loc[primary.state.eq('all_closed_units_collapsed'),'edit_type']='All closed units'
    assert primary.edit_type.notna().all()
    primary['is_outlier']=primary.outlier_abs_z_gt3.astype(bool)
    primary['opposite_sign']=primary.delta_J_minutes*primary.centroid_shift_minutes<0
    primary['group']='Not flagged'
    primary.loc[primary.is_outlier&primary.opposite_sign,'group']='G3 opposite-sign local'
    primary.loc[primary.is_outlier&~primary.opposite_sign&primary.edit_type.eq('Compound member'),'group']='G2 same-sign member'
    primary.loc[primary.is_outlier&~primary.opposite_sign&primary.edit_type.isin(['All closed units','Full compound']),'group']='G1 same-sign aggregate'
    assert primary[primary.is_outlier].group.value_counts().to_dict()=={'G1 same-sign aggregate':11,'G3 opposite-sign local':10,'G2 same-sign member':4}
    # Compute analytic mean and second central moment for the entire comparison population.
    rows=[]
    for ex,g in primary.groupby('example'):
        for state in ['S000_base',*g.state.unique()]:
            f=edge_metrics(read_saved(ex,state));rows.append(dict(example=ex,state=state,**state_moments(f)))
        print('Analytic moments',ex,flush=True)
    sm=pd.DataFrame(rows);sm.to_csv(DEST/'state_moments.csv',index=False)
    parents=sm[sm.state.eq('S000_base')].drop(columns='state').rename(columns={c:'parent_'+c for c in sm.columns if c not in ['example','state']})
    d=primary.merge(sm,on=['example','state'],validate='one_to_one').merge(parents,on='example',validate='many_to_one')
    np.testing.assert_allclose(d.mean_min-d.parent_mean_min,d.delta_J_minutes,rtol=1e-8,atol=1e-7)
    for col in ['variance_h2','path_variance_h2','diffusion_variance_h2','transfer_sd_hours']:
        d['delta_'+col]=d[col]-d['parent_'+col]
    d['sd_ratio']=d.transfer_sd_hours/d.parent_transfer_sd_hours
    d['diffusion_variance_ratio']=d.diffusion_variance_h2/d.parent_diffusion_variance_h2
    # Reference network line sensitivity is reported, not refitted or used to select controls.
    sens=pd.read_csv(OUT/'patterns/window_translation_sensitivity.csv',dtype={'example':str})
    d=d.merge(sens[sens.window.eq('0.5%')][['example','window_translation_sensitivity']],on='example',validate='many_to_one')
    pooled=all_data[(all_data.window=='0.5%')].iloc[0]
    score=pd.read_csv(OUT/'depth_methods/scores.csv');score=score[(score.method=='manning_midpoint')&(score.window=='0.5%')].iloc[0]
    d['network_window_linear_residual_min']=d.centroid_shift_minutes-d.window_translation_sensitivity*d.delta_J_minutes
    d['delta_sigma_squared_hours2']=d.delta_variance_h2
    # Matched descriptive controls: same network/type/sign and nearest |ΔJ| on log scale, never matched on hydraulic variance.
    control_rows=[]
    for r in d[d.is_outlier].itertuples():
        pool=d[(~d.is_outlier)&(d.example==r.example)&(d.edit_type==r.edit_type)&(np.sign(d.delta_J_minutes)==np.sign(r.delta_J_minutes))].copy()
        tier='same network/type/sign'
        if pool.empty:
            pool=d[(~d.is_outlier)&(d.edit_type==r.edit_type)&(np.sign(d.delta_J_minutes)==np.sign(r.delta_J_minutes))].copy();tier='same type/sign; other network'
        if pool.empty:continue
        pool['distance']=abs(np.log1p(pool.delta_J_minutes.abs())-np.log1p(abs(r.delta_J_minutes)))
        for v in pool.nsmallest(3,'distance').itertuples():
            control_rows.append(dict(outlier_example=r.example,outlier_state=r.state,group=r.group,control_example=v.example,control_state=v.state,
                match_tier=tier,delta_J_min=v.delta_J_minutes,centroid_min=v.centroid_shift_minutes,sd_ratio=v.sd_ratio,delta_variance_h2=v.delta_variance_h2,match_distance=v.distance))
    controls=pd.DataFrame(control_rows);controls.to_csv(DEST/'matched_controls.csv',index=False)
    selected=set(zip(d[d.is_outlier].example,d[d.is_outlier].state))|set(zip(controls.control_example,controls.control_state))
    edges_all=[];changes=[]
    for ex,state in sorted(selected):
        p=edge_metrics(read_saved(ex,'S000_base'));c=edge_metrics(read_saved(ex,state));change,edges=edge_changes(p,c)
        changes.append(dict(example=ex,state=state,**change));edges['example']=ex;edges['state']=state
        # Full detailed edge differences retained for selected cases; no arbitrary edge deletion.
        edges_all.append(edges)
    pd.concat(edges_all,ignore_index=True).to_parquet(DEST/'selected_edge_changes.parquet',index=False)
    changes=pd.DataFrame(changes);changes.to_csv(DEST/'geometry_changes.csv',index=False)
    d=d.merge(changes,on=['example','state'],how='left',validate='one_to_one')
    # Actual hydrograph accounting for all 25 outliers, using saved curves only.
    waves=[];segment_rows=[]
    @lru_cache(None)
    def wave(ex,state):
        f=read_saved(ex,state,'outlet.parquet');return f.time_utc.astype('int64').to_numpy()/1e9,f.q_outlet_cms.to_numpy()
    for r in d[d.is_outlier].itertuples():
        tp,qp=wave(r.example,'S000_base');tc,qc=wave(r.example,r.state)
        a=pd.Timestamp(r.onset_utc).timestamp();b=pd.Timestamp(r.end_utc).timestamp();center=r.parent_centroid_seconds
        peak=r.parent_peak_seconds
        knots=np.unique(np.r_[a,b,center,peak,tp[(tp>a)&(tp<b)],tc[(tc>a)&(tc<b)]])
        knots=knots[(knots>=a)&(knots<=b)]
        p=np.interp(knots,tp,qp)-r.baseline_cms;c=np.interp(knots,tc,qc)-r.baseline_cms
        mp,vp=moment(knots,p);mc,vc=moment(knots,c)
        np.testing.assert_allclose((mc-mp)/60,r.centroid_shift_minutes,atol=1e-4)
        dm,df=exact_piecewise_integrals(knots,c-p,center)
        for label,mask in [('before_parent_peak',knots[1:]<=peak+1e-5),('after_parent_peak',knots[:-1]>=peak-1e-5),
            ('before_parent_centroid',knots[1:]<=center+1e-5),('after_parent_centroid',knots[:-1]>=center-1e-5)]:
            segment_rows.append(dict(example=r.example,state=r.state,group=r.group,segment=label,delta_volume_m3=dm[mask].sum(),centroid_contribution_min=df[mask].sum()/vc/60))
        np.testing.assert_allclose(df.sum()/vc/60,r.centroid_shift_minutes,atol=1e-4)
        end=min(tp[-1],tc[-1]);tail=np.unique(np.r_[b,tp[(tp>b)&(tp<end)],tc[(tc>b)&(tc<end)],end]);pdiff=np.interp(tail,tc,qc)-np.interp(tail,tp,qp)
        tailmass=np.trapz(pdiff,tail)
        delta=c-p
        waves.append(dict(example=r.example,state=r.state,window_delta_volume_m3=vc-vp,
            post_window_delta_volume_m3=tailmass,end_child_minus_parent_cms=pdiff[-1],
            max_window_child_minus_parent_cms=delta.max(),min_window_child_minus_parent_cms=delta.min(),
            min_difference_time_utc=pd.Timestamp(knots[np.argmin(delta)],unit='s',tz='UTC').isoformat(),
            max_difference_time_utc=pd.Timestamp(knots[np.argmax(delta)],unit='s',tz='UTC').isoformat(),
            saved_tail_hours=(end-b)/3600))
    pd.DataFrame(segment_rows).to_csv(DEST/'hydrograph_segments.csv',index=False)
    wave_table=pd.DataFrame(waves);wave_table.to_csv(DEST/'hydrograph_accounting.csv',index=False)
    # Existing numerical checks: report coverage instead of assuming every state was refined.
    path=SOURCE/'convergence_checks.csv';SOURCES[str(path)]=digest(path)
    checks=pd.read_csv(path,dtype={'example':str});checks=checks[checks.window.eq('0.5%')]
    numerical=[]
    for r in d[d.is_outlier].itertuples():
        c=checks[(checks.example==r.example)&(checks.state==r.state)]
        numerical.append(dict(example=r.example,state=r.state,checks=len(c),controls='|'.join(c.control),
            maximum_centroid_contrast_change_seconds=c.centroid_contrast_change_seconds.abs().max() if len(c) else np.nan,
            all_available_checks_accepted=bool(c.accepted.all()) if len(c) else None))
    numerical=pd.DataFrame(numerical);numerical.to_csv(DEST/'numerical_check_coverage.csv',index=False)
    outliers=d[d.is_outlier].merge(wave_table,on=['example','state'],validate='one_to_one').merge(numerical,on=['example','state'],validate='one_to_one')
    persistence=all_data.merge(outliers[['example','state','group']],on=['example','state'],validate='many_to_one')
    persistence.to_csv(DEST/'outlier_window_persistence.csv',index=False)
    d.to_parquet(DEST/'population_diagnostics.parquet',index=False);outliers.to_csv(DEST/'outlier_diagnostics.csv',index=False)
    summaries=d.groupby('group').agg(n=('state','size'),median_sd_ratio=('sd_ratio','median'),min_sd_ratio=('sd_ratio','min'),max_sd_ratio=('sd_ratio','max'),
        median_delta_variance_h2=('delta_variance_h2','median'),median_abs_residual_min=('ols_residual_minutes',lambda z:z.abs().median()))
    summaries.to_csv(DEST/'group_summary.csv')
    (DEST/'provenance.json').write_text(json.dumps(dict(source_sha256=SOURCES,script_sha256=digest(__file__),
        selection='Fixed Manning, primary 0.5% window, original pooled OLS abs(z)>3; groups 11/4/10 fixed before diagnostics.',
        analytic_moments='Fixed Muskingum transfer: cell mean k, variance k^2(1-2X); reach independent cell variances add; network variance = allocation-weighted local dispersion + variance of route sums. Trapezoidal time transform preserves first two low-frequency moments. Analytic calculation, not new impulse routing.',
        caveat='Transfer central moments characterize the model, not observed event width. Some coefficients can be signed; do not assume every cell impulse is a probability distribution. No causal rerouting interventions, no claim of real-world backwater physics.',
        matching='Nearest log1p absolute ΔJ among unflagged same network/type/sign, up to three; fallback other network same type/sign. Descriptive, not independent samples.'),indent=2)+'\n')
    print(summaries.to_string())
    print(outliers[['example','state','group','delta_J_minutes','centroid_shift_minutes','sd_ratio','delta_variance_h2','delta_path_variance_h2','delta_diffusion_variance_h2','checks','maximum_centroid_contrast_change_seconds']].to_string(index=False))

if __name__=='__main__':main()
