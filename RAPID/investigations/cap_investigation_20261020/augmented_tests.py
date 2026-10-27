"""Nested fixed-panel forest tuning and geometric feature screens, isolated outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from edited_size_check import b, OUT, EXT, save, sha
from rapid_tools.current_domains import allocation_tables

ROOT = OUT.parents[2]
DEST = OUT/'augmented_tests'
BASE = list(b.COLUMNS)
TARGET = 'centroid_shift_hours'
PANELS = {
    'baseline': BASE,
    'delta_H': ['delta_H' if c=='log1p__B_ratio_route_entropy' else c for c in BASE],
    'parent_H': BASE+['parent_H'],
    'parent_CV_L': BASE+['parent_CV_L'],
    'parent_width': BASE+['parent_width_km'],
    'parent_slope': BASE+['parent_slope_slow'],
    'width_and_slope': BASE+['parent_width_km', 'parent_slope_slow'],
    'length_delta': BASE+['length_delta_km'],
}


def prepare():
    sources = {}
    def read(path):
        path = Path(path)
        sources[str(path.resolve())] = sha(path)
        if path.suffix == '.json':
            return json.loads(path.read_text())
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        return pd.read_csv(path, dtype={'example':str})
    d = read(OUT/'edited_size_check/uncapped_edited_input.parquet')
    scopes = ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
    m = read(scopes/'geometry_responses.parquet')
    e = read(EXT/'edited_predictors.parquet')
    cols = ['example', 'state', 'B_parent_route_entropy', 'B_child_route_entropy',
            'B_parent_route_length_sd', 'B_parent_route_length_mean', 'B_child_route_length_mean',
            'B_parent_width_mean']
    src = pd.concat([m[cols], e[cols]], ignore_index=True)
    assert not src.duplicated(['example','state']).any()
    d = d.merge(src, on=['example','state'], how='left', validate='one_to_one')
    assert d[cols[2:]].notna().all().all()
    d['delta_H'] = d.B_child_route_entropy-d.B_parent_route_entropy
    d['parent_H'] = d.B_parent_route_entropy
    d['parent_CV_L'] = d.B_parent_route_length_sd/d.B_parent_route_length_mean
    d['parent_width_km'] = d.B_parent_width_mean/1000
    d['length_delta_km'] = (d.B_child_route_length_mean-d.B_parent_route_length_mean)/1000
    graphs = read(scopes/'domain_graphs.csv')
    plans = [read(EXT/'routing_plan.json'), read(OUT/'routing_plan.json')]
    cases = {c['example']:c for plan in plans for c in plan['cases']}
    parents = {}
    for ex, case in cases.items():
        job = next(j for j in case['jobs'] if j['state_id']=='S000_base')
        parents[ex] = read(job['parameters']).set_index('id_link')
    scope_meta = m.set_index(['example','state'])
    cache, slopes = {}, []
    for row in d.itertuples():
        full = parents[row.example]
        if row.cohort == 'segmented':
            lineage = read(EXT/f'variants/{row.example}/all_closed_units_collapsed/lineage.csv')
            ids = lineage.loc[lineage.role.eq('collapsed_block'), 'parent_link_id'].unique()
            selected = full.loc[ids]
            weights = selected.reference_flow_fraction*selected.link_length_m
        else:
            scope = scope_meta.loc[(row.example,row.state), 'B_scope']
            g = graphs[graphs.example.eq(row.example)&graphs.state.eq(row.state)&graphs.scope.eq(scope)]
            assert len(g)==1
            key = g.iloc[0].parent_graph
            if key not in cache:
                geom = read(scopes/'graphs'/f'{key}.parquet')
                cache[key] = allocation_tables(geom)[1].set_index('id_link')
            edges = cache[key]
            selected = full.loc[edges.index]
            np.testing.assert_allclose(selected.link_length_m, edges.L_e_m, rtol=1e-10, atol=1e-7)
            weights = edges.mean_length_contribution_m
        assert (selected.slope>0).all() and weights.sum()>0
        np.testing.assert_allclose(weights.sum(), row.B_parent_route_length_mean, rtol=1e-10, atol=1e-6)
        slopes.append(float(np.average(1/np.sqrt(selected.slope), weights=weights)))
    d['parent_slope_slow'] = slopes
    assert len(d)==2185 and d.example.nunique()==15
    assert np.isfinite(d[list(dict.fromkeys(sum(PANELS.values(), [])))].to_numpy()).all()
    d.to_parquet(DEST/'input.parquet', index=False)
    params = read(b.OUT/'final_model.json')['params']
    candidates = [
        dict(name='frozen', params=params),
        dict(name='split2', params=dict(params, min_samples_split=2)),
        dict(name='fine_absolute', params=dict(params, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.)),
        dict(name='fine_squared', params=dict(params, criterion='squared_error', min_samples_split=2, min_samples_leaf=2, ccp_alpha=0.)),
    ]
    for path in [Path(__file__), Path(b.__file__), Path(b.forest.__file__), Path(b.engine.__file__)]:
        sources[str(path.resolve())] = sha(path)
    design = dict(sources=sources, panels=PANELS, candidates=candidates,
        target=TARGET, seed=b.engine.SEED, primary_window='0.5%', rows=len(d),
        training='Original plus segmented, no pairs. Equal network-total sample weights. Training-only preprocessing and target standardization.',
        tuning='15 outer held-network folds; 3 grouped inner folds select one of four forest configurations using baseline seven inputs only.',
        objective='Mean of original and segmented cohort relative errors; denominator floor 1 minute, equal network weight within each cohort.',
        feature_screen='Use that same training-selected configuration for every feature panel in each outer fold. No feature-panel selection from test outcomes. Panels are exploratory screens, not separately retuned models.',
        domains='S_L is edited size. Other parent context and length delta use the SAME descriptor domain as the existing ratios: outer for original, edited for segmented. Length delta is child mean minus parent mean, not edited size times outer ratio.',
        slope='Visit-length-weighted mean of 1/sqrt(geometric slope); no depth, roughness, celerity, K/X or response-derived inputs.',
        limitations=['Previously examined networks, not an independent external test.', 'Only 15 independent segmented cases.',
                     'Subset exclusions are scoring-only; peak scores evaluate centroid-trained predictions.'],
        production_modified=False)
    fp = hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():
        assert json.loads((DEST/'design.json').read_text())==design, 'Design changed; use new output version'
    save(DEST/'design.json',design)
    return d,candidates,fp


def fit(train,params,panel,jobs):
    b.COLUMNS = list(PANELS[panel])
    return b.fit(train,params,jobs)


def objective(frame,p):
    values = []
    for cohort in ['original','segmented']:
        mask = frame.cohort.eq(cohort).to_numpy()
        z = frame.loc[mask]
        assert len(z)
        values.append(b.forest.sweep.relative_score(z[TARGET].to_numpy(),p[mask],b.engine.weights(z),1))
    return float(np.mean(values)), values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    d,candidates,fp=prepare()
    print('Prepared: 180 inner fits + 120 outer panel fits; 15 held-out networks. No production changes.',flush=True)
    if not args.run:
        return
    predictions,selections,trials = [],[],[]
    for ex in sorted(d.example.unique()):
        train,test = d[d.example.ne(ex)].reset_index(drop=True), d[d.example.eq(ex)]
        ranked = []
        folds = list(GroupKFold(3).split(train,groups=train.example))
        for candidate in candidates:
            path=DEST/'inner'/ex/f'{candidate["name"]}.json'
            result=b.forest.checked(path,fp)
            if result is None:
                pred=np.full(len(train),np.nan)
                for ti,vi in folds:
                    tr,va=train.iloc[ti],train.iloc[vi]
                    assert set(tr.example).isdisjoint(va.example) and ex not in set(tr.example)
                    artifact=fit(tr,candidate['params'],'baseline',args.jobs)
                    pred[vi]=b.forest.predict(artifact,va)
                assert np.isfinite(pred).all()
                score,by=objective(train,pred)
                result=dict(fingerprint=fp,name=candidate['name'],params=candidate['params'],score=score,cohort_scores=by)
                save(path,result)
            ranked.append(result);trials.append(dict(held=ex,**result))
            print(ex,candidate['name'],'inner score',round(result['score'],2),flush=True)
        best=min(enumerate(ranked),key=lambda v:(v[1]['score'],v[0]))[1]
        selections.append(dict(held=ex,**best))
        for panel in PANELS:
            path=DEST/'outer'/ex/f'{panel}.json'
            result=b.forest.checked(path,fp)
            if result is None:
                artifact=fit(train,best['params'],panel,args.jobs)
                pred=b.forest.predict(artifact,test)
                result=dict(fingerprint=fp,selected=best['name'],rows=[
                    dict(example=r.example,state=r.state,cohort=r.cohort,centroid_shift_hours=r.centroid_shift_hours,
                         peak_shift_hours=r.peak_shift_hours,predicted_hours=float(v)) for r,v in zip(test.itertuples(),pred)])
                save(path,result)
            predictions.extend(dict(panel=panel,**r) for r in result['rows'])
        save(DEST/'selections.json',selections)
        save(DEST/'inner_trials.json',trials)
        print('Held-out',ex,'complete; selected',best['name'],flush=True)
    b.COLUMNS=BASE
    p=pd.DataFrame(predictions)
    assert len(p)==len(d)*len(PANELS) and not p.duplicated(['panel','example','state']).any()
    frozen=pd.read_parquet(OUT/'edited_size_check/predictions.parquet')
    frozen=frozen[frozen.version.eq('uncapped')&frozen['size'].eq('edited')&frozen.variant.eq('plus_segmented')].copy()
    frozen['panel']='frozen_reference'
    p=pd.concat([p,frozen[p.columns]],ignore_index=True)
    p.to_parquet(DEST/'predictions.parquet',index=False)
    scores=[]
    for (panel,cohort),g in p.groupby(['panel','cohort']):
        for target in [TARGET,'peak_shift_hours']:
            for subset,z in [('all',g),('excluding54',g[g.example.ne('54')]),('excluding22',g[g.example.ne('22')])]:
                y,pr,w=z[target].to_numpy(),z.predicted_hours.to_numpy(),b.engine.weights(z)
                scores.append(dict(panel=panel,cohort=cohort,target=target,subset=subset,n=len(z),
                    relative_error_pct=b.forest.sweep.relative_score(y,pr,w,1),**b.engine.score(y*60,pr*60,w)))
    s=pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv',index=False)
    design=json.loads((DEST/'design.json').read_text())
    for path,h in design['sources'].items():
        assert sha(path)==h
    save(DEST/'completion.json',dict(complete=True,fingerprint=fp,inner_fits=180,outer_fits=120,production_modified=False))
    print(s.query("target=='centroid_shift_hours' and subset=='all'").to_string(index=False))


if __name__=='__main__':
    main()
