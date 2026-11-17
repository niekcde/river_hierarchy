"""ΔJ by saved depth method, with descriptive OLS residual z-score screening.

Reads completed routing only. BASED evaluates the frozen Manning tangent at
prescribed depth; it does not assert normal-depth storage consistency at Qref.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from incremental_storage import ROOT, SOURCE, OUT, digest

DEST=OUT/'depth_methods'
METHODS={
    'manning_midpoint':('03_manning_midpoint','Fixed Manning — midpoint Q'),
    'based_midpoint':('02_based_midpoint','Fixed BASED — midpoint Q'),
    'based_original':('01_based_slope_geometry_correction','Fixed BASED — original Q reference'),
}


def build():
    sources={};tables=[];state_tables=[];summaries=[]
    for method,(run,label) in METHODS.items():
        source=SOURCE.parent/run
        path=source/'responses.csv';receipt=json.loads((source/'completion.json').read_text())
        sha=digest(path);assert sha==receipt['files'][str(path)];sources[str(path)]=sha
        responses=pd.read_csv(path,dtype={'example':str})
        responses=responses[responses.state.ne('fully_collapsed')].copy()
        rows=[]
        for example,g in responses.groupby('example'):
            for state in ['S000_base',*sorted(g.state.unique())]:
                path=source/'states'/example/state/'parameters.parquet'
                rec=json.loads((path.parent/'completion.json').read_text());sha=digest(path)
                assert sha==rec['files'][str(path)],str(path);sources[str(path)]=sha
                f=pd.read_parquet(path)
                cols=['reference_flow_fraction','link_length_m','width','depth_m','slope','n_manning','reference_discharge_cms']
                assert np.isfinite(f[cols]).all().all() and (f[cols]>0).all().all()
                assert f.n_manning.nunique()==1
                qref=f.reference_discharge_cms/f.reference_flow_fraction
                np.testing.assert_allclose(qref,qref.iloc[0],rtol=1e-10)
                np.testing.assert_allclose(f['count']*f.rapid_effective_length_m,f.link_length_m,rtol=1e-10)
                craw=5/(3*f.n_manning)*f.depth_m**(2/3)*np.sqrt(f.slope)
                np.testing.assert_allclose(craw,f.rapid_celerity_mps_raw,rtol=1e-10)
                j=float((f.reference_flow_fraction*f.link_length_m/craw).sum())
                routed=float((f.reference_flow_fraction*f['count']*f.rapid_k).sum())
                np.testing.assert_allclose(routed,(f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps).sum(),rtol=1e-10)
                capped=int(f.rapid_celerity_capped.sum())
                if not capped:np.testing.assert_allclose(j,routed,rtol=1e-10)
                if method=='manning_midpoint':
                    normal=(f.n_manning*f.reference_discharge_cms/(f.width*np.sqrt(f.slope)))**.6
                    np.testing.assert_allclose(f.depth_m,normal,rtol=1e-10)
                    np.testing.assert_allclose(j,.6*(f.width*f.link_length_m*f.depth_m).sum()/qref.iloc[0],rtol=1e-10)
                rows.append(dict(example=example,state=state,J_seconds=j,routed_first_moment_seconds=routed,
                    capped_reaches=capped,n=float(f.n_manning.iloc[0]),Qref_cms=float(qref.iloc[0])))
            print(method,example,flush=True)
        states=pd.DataFrame(rows)
        parents=states[states.state.eq('S000_base')].drop(columns='state').rename(columns={c:'parent_'+c for c in states.columns if c not in ['example','state']})
        d=responses.merge(states,on=['example','state'],validate='many_to_one').merge(parents,on='example',validate='many_to_one')
        assert len(d)==len(responses)
        np.testing.assert_allclose(d.n,d.parent_n,rtol=1e-12)
        np.testing.assert_allclose(d.Qref_cms,d.parent_Qref_cms,rtol=1e-12)
        d['method']=method;d['method_label']=label;states['method']=method
        d['delta_J_minutes']=(d.J_seconds-d.parent_J_seconds)/60
        d['delta_routed_mean_minutes']=(d.routed_first_moment_seconds-d.parent_routed_first_moment_seconds)/60
        d['celerity_cap_effect_minutes']=d.delta_routed_mean_minutes-d.delta_J_minutes
        d['centroid_shift_minutes']=d.centroid_shift_hours*60
        d['identity_residual_minutes']=d.centroid_shift_minutes-d.delta_J_minutes
        for window,indices in d.groupby('window').groups.items():
            g=d.loc[indices];x=g.delta_J_minutes.to_numpy();y=g.centroid_shift_minutes.to_numpy()
            slope,intercept=np.polyfit(x,y,1);fitted=slope*x+intercept;res=y-fitted
            sd=float(np.std(res,ddof=1));z=(res-res.mean())/sd if sd>0 else np.full(len(res),np.nan)
            d.loc[indices,'ols_fitted_minutes']=fitted;d.loc[indices,'ols_residual_minutes']=res
            d.loc[indices,'residual_z']=z;d.loc[indices,'outlier_abs_z_gt3']=np.abs(z)>3
            den=float(((y-y.mean())**2).sum())
            summaries.append(dict(method=method,window=window,n=len(g),networks=g.example.nunique(),
                ols_slope=slope,ols_intercept_minutes=intercept,ols_r2=1-float((res**2).sum())/den,
                residual_sd_minutes=sd,ols_mae_minutes=float(abs(res).mean()),outliers_abs_z_gt3=int((abs(z)>3).sum()),
                identity_r2=1-float(((y-x)**2).sum())/den,identity_mae_minutes=float(abs(y-x).mean()),
                comparisons_with_capped_reaches=int(((g.capped_reaches+g.parent_capped_reaches)>0).sum()),
                max_abs_cap_effect_minutes=float(g.celerity_cap_effect_minutes.abs().max())))
        tables.append(d);state_tables.append(states)
    data=pd.concat(tables,ignore_index=True);states=pd.concat(state_tables,ignore_index=True);summary=pd.DataFrame(summaries)
    # Require identical parent-child/window populations across methods, never a cross-run response join.
    keys=['example','state','window'];ref=tables[0].set_index(keys).index.sort_values()
    for d in tables[1:]:assert d.set_index(keys).index.sort_values().equals(ref)
    original=pd.read_parquet(OUT/'comparisons.parquet').sort_values(keys)
    check=data[data.method.eq('manning_midpoint')].sort_values(keys)
    np.testing.assert_allclose(check.delta_J_minutes,original.delta_J_minutes,rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(check.centroid_shift_minutes,original.centroid_shift_minutes,rtol=1e-12)
    DEST.mkdir(exist_ok=True,parents=True)
    data.to_parquet(DEST/'comparisons.parquet',index=False);states.to_csv(DEST/'state_storage.csv',index=False)
    summary.to_csv(DEST/'scores.csv',index=False)
    data[data.outlier_abs_z_gt3.astype(bool)].to_csv(DEST/'zscore_outliers.csv',index=False)
    (DEST/'provenance.json').write_text(json.dumps(dict(methods=METHODS,source_sha256=sources,script_sha256=digest(__file__),
        definition='ΔJ=sum f*3*n*L/(5*h^(2/3)*sqrt(S)), child minus parent; use each run own depths/n and responses.',
        based_interpretation='Frozen Manning tangent at prescribed BASED depth, not a claim of normal-flow storage consistency at allocated Qref. Celerity-cap effect audited separately.',
        residual_screen='OLS with intercept per method and event window, all networks pooled. z=(residual-mean)/sample SD, ddof=1; abs(z)>3. Descriptive, correlated comparisons, no significance claim. Not a production predictor fit.',
        scopes='All comparisons except fully_collapsed single-centerline control; existing Manning pattern sections unchanged.'),indent=2)+'\n')
    print(summary[summary.window.eq('0.5%')].to_string(index=False))
    return data

if __name__=='__main__':build()
