"""QA and matched previous/selected summaries; run after routing and benchmarks."""
from pathlib import Path
import json,hashlib,shutil
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'synthetic_experiments/outputs'
V=OLD/'hydraulic_alpha065_20260930'
Y=['lag_shift_pct','event_duration_change_pct','peak_excess_change_pct']

def main():
    quality=[]
    for dataset in ['full_grid_island','embedded_width_length_extension']:
        out=V/dataset;m=json.loads((out/'manifest.json').read_text());assert m['status']=='analysis_complete'
        stats=pd.DataFrame([json.loads((out/'runs'/rid/'metrics.json').read_text()) for rid in m['run_ids']])
        assert max(stats.volume_error_fraction.abs().max(),stats.pre_event_relative_error.max(),stats.final_relative_error.max())<1e-4
        quality.append(dict(dataset=dataset,configurations=m['n_configurations'],comparisons=m['n_comparisons'],unique_runs=m['n_unique_runs'],depth_alpha=.65,n_manning=.035,max_volume_error=stats.volume_error_fraction.abs().max(),max_pre_event_error=stats.pre_event_relative_error.max(),max_final_error=stats.final_relative_error.max(),capped_runs=int((stats.n_celerity_capped>0).sum()),floored_runs=int((stats.n_length_floored>0).sum())))
    pd.DataFrame(quality).to_csv(V/'quality_summary.csv',index=False)
    old=pd.read_parquet(OLD/'full_grid_island/current_analysis/current_comparisons.parquet');new=pd.read_parquet(V/'full_grid_island/current_analysis/current_comparisons.parquet')
    keys=['configuration_id','child_state'];a=old.merge(new,on=keys,suffixes=('_old','_new'),validate='one_to_one');assert len(a)==15309
    for c in ['geometry_group','split','development_fold']:
        assert np.array_equal(a[c+'_old'],a[c+'_new']),c
    reuse=json.loads((V/'full_grid_island/current_analysis/geometry_reuse.json').read_text())
    for c in reuse['columns']:
        if c in keys or c in ['geometry_group','split','development_fold']:continue
        assert np.allclose(a[c+'_old'],a[c+'_new'],equal_nan=True),c
    changes=[]
    for state,g in a.groupby('child_state'):
        for y in Y+['resistance_predictor']:
            delta=g[y+'_new']-g[y+'_old']
            changes.append(dict(child_state=state,variable=y,n=len(g),old_median=g[y+'_old'].median(),selected_median=g[y+'_new'].median(),median_signed_difference=delta.median(),median_absolute_difference=delta.abs().median(),maximum_absolute_difference=delta.abs().max(),sign_changes=int((np.sign(g[y+'_new'])!=np.sign(g[y+'_old'])).sum())))
    pd.DataFrame(changes).to_csv(V/'old_new_response_summary.csv',index=False)
    scores=[]
    for label,root in [('previous',OLD),('selected',V)]:
        d=pd.read_csv(root/'full_grid_island/current_analysis/predictor_scores.csv')
        d=d[(d.validation=='geometry_group_holdout')&(d.child_state!='pooled')]
        q=d.groupby(['model','feature_set','response'])[['r2','mae_pp']].mean().reset_index();q['hydraulic_setup']=label;scores.append(q)
    scores=pd.concat(scores);scores.to_csv(V/'old_new_predictor_long.csv',index=False)
    p=scores.pivot(index=['model','feature_set','response'],columns='hydraulic_setup',values=['r2','mae_pp']);p.columns=['_'.join(c) for c in p.columns];p['r2_change']=p.r2_selected-p.r2_previous;p.reset_index().to_csv(V/'old_new_predictor_summary.csv',index=False)
    # Match the prior focused diagnostic at identical parameters and clock policy.
    focused=pd.read_csv(OLD/'width_depth_scaling/results.csv').query("alpha==.65 and clock_mode=='native_clock'")
    matchcols=['child_state','total_width_m','reference_slope','outer_a_fraction','inner_fraction','length_multiplier','total_length_m','internal_location']
    q=new.merge(focused,on=matchcols,suffixes=('_grid','_focused'),validate='one_to_one')
    assert len(q)==35
    for y in Y+['resistance_predictor']:assert np.allclose(q[y+'_grid'],q[y+'_focused'],rtol=1e-7,atol=1e-7),y
    # Export the selected extension's simple parents for existing illustration cells.
    ext=pd.read_csv(V/'embedded_width_length_extension/full_results.csv');ill=V/'side_channel_width_length';ill.mkdir(exist_ok=True)
    for _,r in ext[(ext.network_extent_m==20000)&(ext.child_state=='simple/full')].iterrows():
        key=f'a{r.outer_a_fraction:g}_m{r.length_multiplier:g}';folder=V/'embedded_width_length_extension/runs'/r.parent_run_id
        shutil.copy2(folder/'hydrograph.parquet',ill/f'{key}_hydrograph.parquet');shutil.copy2(folder/'metrics.json',ill/f'{key}_metrics.json')
    # Data used by the extended geometry table; preserve exact pair identities.
    ext.to_csv(V/'embedded_width_length_extension/plan.csv',index=False)
    provenance=dict(reference='SYN-2026-09-30-SELECTED-RERUN',selected_alpha=.65,focused_parity_comparisons=len(q),frozen_geometry_and_partitions_verified=True,original_grid_comparisons=len(a),quality=quality,
        contrast='Full hydraulic setup change: kb20/n.35 versus BASED reference/local alpha.65/n.035; not an alpha-only intervention.',real_case_decision='deferred pending new updated-slope runs',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (V/'validation.json').write_text(json.dumps(provenance,indent=2));print(pd.DataFrame(quality).to_string(index=False));print('35 focused matches; geometry and validation splits preserved.')
if __name__=='__main__':main()
