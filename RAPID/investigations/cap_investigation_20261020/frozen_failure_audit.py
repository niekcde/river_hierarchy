"""Read-only audit of frozen saved LOEO predictions; never imports fitting code.

Run with a pandas/pyarrow environment. Writes only its own diagnostic outputs.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'RAPID/outputs/uncapped_event_window_20261020'
EXT = ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
DEST = OUT/'frozen_failure_audit_20261026'
SOURCES = {}
KEY = ['example', 'state']
TARGET = 'centroid_shift_hours'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p = Path(p)
    SOURCES[str(p)] = sha(p)
    return json.loads(p.read_text()) if p.suffix == '.json' else pd.read_parquet(p)


def save(name, value):
    (DEST/name).write_text(json.dumps(value, indent=2)+'\n')


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'manifest.json').exists():
        for path, digest in json.loads((DEST/'manifest.json').read_text())['protected_sha256'].items():
            assert sha(path) == digest, 'Frozen source changed: '+path
    settings = read(OUT/'current_model_comparison_20261023/settings.json')
    protected = {str(OUT/'current_model_comparison_20261023/settings.json'): sha(OUT/'current_model_comparison_20261023/settings.json')}
    d = read(OUT/'resistance_inverse_depth_panels_20261023/input.parquet').sort_values(KEY).reset_index(drop=True)
    nine = read(OUT/'augmented_tests/input.parquet').sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(d[nine.columns], nine)
    assert len(d) == 2185 and not d.duplicated(KEY).any()
    assert d.groupby('cohort').size().to_dict() == {'original': 2170, 'segmented': 15}
    assert d.loc[d.cohort.eq('segmented'), 'state'].eq('all_closed_units_collapsed').all()
    datasets, folds, scores, selected, support, controls, neighbors, feature_ranges = {}, [], [], [], [], [], [], []
    historical_checks = []
    for model, folder, variant, meta in [
        ('geometry_9', 'nine_input_tuning', 'same_grid', 'research_model_same_grid.json'),
        ('depth_informed_10', 'ten_input_tuning_20261023', 'nested_tuned', 'final_model.json')]:
        spec = settings[model]; cols = spec['columns']; directory = OUT/folder
        metadata = read(directory/meta)
        artifact = Path(spec['artifact']); protected[str(artifact)] = sha(artifact)
        assert protected[str(artifact)] == metadata['sha256']
        assert spec['columns'] == metadata['columns'] and spec['params'] == metadata['selected']['params']
        design = read(directory/'design.json')
        assert spec['seed'] == design['seed']
        for path, digest in design['sources'].items():
            historical_checks.append(dict(model=model,path=path,exists=Path(path).exists(),matches=Path(path).exists() and sha(path)==digest))
        protected[str(directory/'predictions.parquet')] = sha(directory/'predictions.parquet')
        pred = read(directory/'predictions.parquet'); pred = pred[pred.variant.eq(variant)].sort_values(KEY).reset_index(drop=True)
        pd.testing.assert_frame_equal(pred[KEY+['cohort',TARGET]], d[KEY+['cohort',TARGET]])
        assert np.isfinite(pred.predicted_hours).all()
        choices = read(directory/'selections.json')
        choices = [c for c in choices if c.get('variant', variant) == variant]
        assert len(choices) == 15 and {c['held'] for c in choices} == set(d.example)
        for c in choices:
            train = set(d.example)-{c['held']}
            for inner in c['inner_groups']:
                assert set(inner['train']).isdisjoint(inner['validation'])
                assert set(inner['train']) | set(inner['validation']) == train
            checkpoint = directory/'outer'/c['held']/f'{c["index"]:02d}.json' if model == 'geometry_9' else directory/'outer'/f'{c["held"]}.json'
            checkpoint_data = read(checkpoint)
            assert checkpoint_data['fingerprint'] == c['fingerprint']
            cp = pd.DataFrame(checkpoint_data['rows']).sort_values(KEY).reset_index(drop=True)
            pp = pred[pred.example.eq(c['held'])].reset_index(drop=True)
            pd.testing.assert_frame_equal(cp[KEY+[TARGET,'predicted_hours']], pp[KEY+[TARGET,'predicted_hours']])
            tr = d[d.example.ne(c['held'])]
            counts = tr.groupby('example').size()
            folds.append(dict(model=model, held=c['held'], selected=c['name'], differs_from_final=c['params']!=spec['params'], training_rows=len(tr), training_networks=14,
                              segmented_training_rows=14, segmented_training_weight_pct=100*np.mean(1/counts)))
        z = d.copy(); z['model'] = model; z['predicted_hours'] = pred.predicted_hours
        z['error_hours'] = z.predicted_hours-z[TARGET]
        z['abs_error_hours'] = abs(z.error_hours)
        z['relative_error_pct'] = 100*z.abs_error_hours/np.maximum(abs(z[TARGET]), 1/60)
        z['success'] = z.abs_error_hours <= np.maximum(1/60, .2*abs(z[TARGET]))
        for cohort, g in z.groupby('cohort'):
            w = 1/g.groupby('example').example.transform('size')
            z.loc[g.index,'sse_share_pct'] = 100*w*g.error_hours**2/np.sum(w*g.error_hours**2)
            z.loc[g.index,'relative_share_pct'] = 100*w*g.relative_error_pct/np.sum(w*g.relative_error_pct)
            scores.append(dict(model=model, cohort=cohort, rows=len(g), networks=g.example.nunique(), r2=1-np.sum(w*g.error_hours**2)/np.sum(w*(g[TARGET]-np.average(g[TARGET],weights=w))**2), mae_minutes=60*np.average(g.abs_error_hours,weights=w), relative_error_pct=np.average(g.relative_error_pct,weights=w), successes=int(g.success.sum())))
        for cohort, g in z.groupby('cohort'):
            for metric in ['sse_share_pct','abs_error_hours','relative_share_pct']:
                top = g.sort_values([metric,'example','state'],ascending=[False,True,True]).head(5).copy()
                top['criterion'] = metric; top['rank'] = np.arange(1,len(top)+1)
                selected.append(top)
        # Case investigation: top three SSE/absolute and top two relative contributions.
        ids = set()
        for _, g in z.groupby('cohort'):
            for metric in ['sse_share_pct','abs_error_hours','relative_share_pct']:
                count = 2 if metric == 'relative_share_pct' else 3
                ids.update(g.sort_values([metric,'example','state'],ascending=[False,True,True]).head(count).index)
        for i in sorted(ids):
            r = z.loc[i]; tr = z[z.example.ne(r.example)].copy()
            scale = tr[cols].quantile(.75)-tr[cols].quantile(.25)
            scale = scale.where(scale>1e-12, tr[cols].std()).fillna(1).replace(0,1)
            distance = np.sqrt((((z[cols]-r[cols].astype(float))/scale)**2).mean(axis=1))
            tr['distance'] = distance.loc[tr.index]
            closest = tr.sort_values(['distance','example','state']).groupby('example',sort=False).head(1).head(5)
            for rank, (_, n) in enumerate(closest.iterrows(),1):
                neighbors.append(dict(model=model, example=r.example, state=r.state, rank=rank, neighbor_example=n.example, neighbor_state=n.state, cohort=n.cohort, distance=n.distance, target_hours=n[TARGET], error_hours=n.error_hours))
            outside = []
            for col in cols:
                lo, hi = tr[col].min(), tr[col].max()
                is_out = r[col]<lo-1e-10 or r[col]>hi+1e-10
                if is_out: outside.append(col)
                feature_ranges.append(dict(model=model, example=r.example, state=r.state, feature=col,value=r[col],train_min=lo,train_max=hi,outside=bool(is_out),nearest_network_value=closest.iloc[0][col]))
            comparable = tr[(tr.cohort==r.cohort)&(np.sign(tr[TARGET])==np.sign(r[TARGET]))&(abs(tr[TARGET])>=abs(r[TARGET])/2)&(abs(tr[TARGET])<=2*abs(r[TARGET]))]
            support.append(dict(model=model,example=r.example,state=r.state,cohort=r.cohort,target_hours=r[TARGET],predicted_hours=r.predicted_hours,error_hours=r.error_hours,sse_share_pct=r.sse_share_pct,
                train_target_min=tr[TARGET].min(),train_target_max=tr[TARGET].max(),outside_target=bool(r[TARGET]<tr[TARGET].min() or r[TARGET]>tr[TARGET].max()),outside_features=' | '.join(outside),nearest_distance=closest.iloc[0].distance,comparable_rows=len(comparable),comparable_networks=comparable.example.nunique()))
            eligible = tr[(tr.cohort==r.cohort)&tr.success&(np.sign(tr[TARGET])==np.sign(r[TARGET]))].copy()
            eligible['magnitude_ratio'] = abs(eligible[TARGET])/abs(r[TARGET])
            matched = eligible[eligible.magnitude_ratio.between(.5,2)]
            pool = matched if len(matched) else eligible
            if len(pool):
                n = pool.sort_values(['distance','example','state']).iloc[0]
                controls.append(dict(model=model,example=r.example,state=r.state,cohort=r.cohort,control_example=n.example,control_state=n.state,target_hours=r[TARGET],error_hours=r.error_hours,control_target_hours=n[TARGET],control_prediction_hours=n.predicted_hours,control_error_hours=n.error_hours,distance=n.distance,magnitude_ratio=n.magnitude_ratio,magnitude_matched=bool(len(matched))))
        datasets[model] = z
    pd.DataFrame(historical_checks).to_csv(DEST/'historical_source_checks.csv',index=False)
    pd.concat(datasets.values()).to_parquet(DEST/'all_errors.parquet',index=False)
    pd.concat(selected).to_csv(DEST/'rankings.csv',index=False)
    for name, rows in [('scores',scores),('fold_audit',folds),('case_support',support),('matched_successes',controls),('nearest_networks',neighbors),('feature_ranges',feature_ranges)]:
        pd.DataFrame(rows).to_csv(DEST/f'{name}.csv',index=False)
    # Independently reconstruct response lineage, including unchanged networks.
    replaced = read(OUT/'review/uncapped_responses.parquet')
    original = read(EXT/'responses.parquet')
    responses = pd.concat([original[~original.example.isin(replaced.example)],replaced])
    primary = responses[responses.window.eq('0.5%')]
    linked = d[KEY+[TARGET]].merge(primary[KEY+[TARGET]],on=KEY,validate='one_to_one',suffixes=('_input','_response'))
    assert len(linked)==len(d)
    np.testing.assert_array_equal(linked[TARGET+'_input'],linked[TARGET+'_response'])
    new_states = read(OUT/'review/uncapped_state_metrics.parquet')
    old_states = read(EXT/'state_metrics.parquet')
    states = pd.concat([old_states[~old_states.example.isin(new_states.example)],new_states])
    # Raw-wave check uses independent three-point Gauss integration (exact for linear q and t*q).
    keys = {(r['example'],r['state']) for r in support}|{(r['control_example'],r['control_state']) for r in controls}
    wave_checks, sensitivity = [], []
    cache = {}
    def centroid(row):
        path = Path(row.outlet)
        if str(path) not in cache:
            f = read(path); assert SOURCES[str(path)] == row.outlet_sha256
            cache[str(path)] = f
        f = cache[str(path)]; t = f.time_utc.astype('int64').to_numpy()/1e9; q=f.q_outlet_cms.to_numpy()
        a,b = row.rise_seconds,row.fall_seconds
        assert t[0]<=a<b<=t[-1] and np.all(np.diff(t)>0) and np.isfinite(q).all()
        tt=np.r_[a,t[(t>a)&(t<b)],b]; e=np.interp(tt,t,q)-row.baseline_cms
        assert e.min()>=0
        x=tt-a; h=np.diff(x); nodes,weights=np.polynomial.legendre.leggauss(3); u=(nodes+1)/2
        heights=e[:-1,None]+(e[1:]-e[:-1])[:,None]*u
        positions=x[:-1,None]+h[:,None]*u
        volume=np.sum(h[:,None]/2*weights*heights)
        moment=np.sum(h[:,None]/2*weights*heights*positions)
        value=a+moment/volume
        assert abs(value-row.centroid_seconds)<1e-5
        return value
    for ex,state in sorted(keys):
        for window in ['0.5%','1%','2%']:
            rows=states[(states.example==ex)&(states.window==window)].set_index('state')
            child,parent=rows.loc[state],rows.loc['S000_base']
            assert child.rise_seconds==parent.rise_seconds and child.fall_seconds==parent.fall_seconds and child.baseline_cms==parent.baseline_cms
            observed=(centroid(child)-centroid(parent))/3600
            saved=responses[(responses.example==ex)&(responses.state==state)&(responses.window==window)].iloc[0][TARGET]
            assert abs(observed-saved)<1e-8
            wave_checks.append(dict(example=ex,state=state,window=window,recomputed_hours=observed,saved_hours=saved,difference_hours=observed-saved,baseline_cms=child.baseline_cms,start_seconds=child.rise_seconds,end_seconds=child.fall_seconds,child_outlet=child.outlet,parent_outlet=parent.outlet))
        yy=[r['saved_hours'] for r in wave_checks[-3:]]
        sensitivity.append(dict(example=ex,state=state,target_05=yy[0],target_1=yy[1],target_2=yy[2],window_range_hours=max(yy)-min(yy)))
    pd.DataFrame(wave_checks).to_csv(DEST/'raw_wave_checks.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(DEST/'window_sensitivity.csv',index=False)
    # Physical context from actual routing parameter tables; not a new model/input panel.
    jobs = {}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json', EXT/'routing_plan.json', OUT/'routing_plan.json']:
        for case in read(path)['cases']:
            for job in case['jobs']:
                for name in {job['state_id'], *job.get('aliases', [])}:
                    jobs[case['example'], name] = job
    physical = []
    for ex, state in sorted(keys):
        rec = dict(example=ex, state=state)
        for side, name in [('parent','S000_base'),('child',state)]:
            f = read(jobs[ex,name]['parameters'])
            assert f.n_manning.nunique() == 1
            np.testing.assert_allclose(f.celerity_times_n, (5/3)*f.depth_m**(2/3)*np.sqrt(f.slope),rtol=1e-10)
            np.testing.assert_allclose(f.rapid_celerity_mps, np.clip(f.celerity_times_n/f.n_manning,f.celerity_min_mps,f.celerity_max_mps),rtol=1e-10)
            a = f.depth_m/f.width**.65
            assert a.std()/a.mean()<1e-10
            rec[side+'_n'] = float(f.n_manning.iloc[0])
            rec[side+'_depth_anchor'] = float(a.mean())
            rec[side+'_clipped_links'] = int((abs(f.rapid_celerity_mps-f.celerity_times_n/f.n_manning)>1e-10).sum())
            rec[side+'_travel_proxy_hours'] = float((f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps).sum()/3600)
        rec['delta_travel_proxy_hours'] = rec['child_travel_proxy_hours']-rec['parent_travel_proxy_hours']
        rr = primary[(primary.example==ex)&(primary.state==state)].iloc[0]
        for col in [TARGET,'peak_shift_hours','volume_change_pct','parent_asymmetry','child_asymmetry','parent_duration_hours']:
            rec[col] = float(rr[col])
        rec['centroid_minus_proxy_hours'] = rec[TARGET]-rec['delta_travel_proxy_hours']
        physical.append(rec)
    pd.DataFrame(physical).to_csv(DEST/'physical_context.csv',index=False)
    # Hash verification of the previous comparison manifest is a check, not evidence for explanations.
    previous=read(OUT/'current_model_comparison_20261023/manifest.json')
    prior_checks=[dict(path=p,exists=Path(p).exists(),matches=Path(p).exists() and sha(p)==h) for p,h in previous['sources'].items()]
    for p,h in {**protected,**SOURCES}.items(): assert sha(p)==h,p
    SOURCES[str(Path(__file__))]=sha(__file__)
    save('manifest.json',dict(date='2026-10-26',fits=0,settings=settings,protected_sha256=protected,sources=SOURCES,previous_manifest_checks=prior_checks,
        target_join_rows=len(linked),raw_wave_cases=len(keys),raw_wave_windows=len(wave_checks),
        selection='Top five per model/cohort for absolute error, equal-network SSE share and equal-network floored relative-error share; case investigation union of top three absolute, top three SSE and top two relative contributions.',
        success='abs(error) <= max(1/60 hour, 0.2*abs(target)); other network, same cohort and sign; prefer target magnitude ratio [0.5,2], otherwise explicitly mark unmatched magnitude.',
        distance='RMS feature distance divided by outer-training IQR; fallback training SD then 1 for constant feature. One nearest row per distinct training network. Descriptive, not forest leaf similarity.',
        caveat='Saved nested procedure predictions, not predictions of identical fixed all-data hyperparameters; reused networks are not independent external validation. Controls selected post hoc using observed outcomes.'))
    print(pd.DataFrame(scores).to_string(index=False))
    print(pd.DataFrame(support).to_string(index=False))
    print(pd.DataFrame(controls).to_string(index=False))


if __name__ == '__main__':
    main()
