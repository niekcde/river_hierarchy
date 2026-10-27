"""Fixed-settings addition of delta sum(r L / (h sqrt(slope)))."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from slope_ratio_audit import ROOT, OUT, EXT, SC, read, SOURCES, validate_domain, allocation_tables
from tune_nine_inputs import b, COLUMNS

DEST = OUT/'inverse_depth_joint_20261023'
FEATURE = 'delta_G_inverse_depth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST/'preparation.json').exists():
        m = json.loads((DEST/'preparation.json').read_text())
        assert m['script_hash'] == sha(__file__)
        for path, h in m['sources'].items():
            assert sha(path) == h, path
        assert sha(DEST/'input.parquet') == m['input_hash']
        return pd.read_parquet(DEST/'input.parquet'), m
    d = read(OUT/'augmented_tests/input.parquet')
    meta = read(SC/'geometry_responses.parquet').set_index(['example','state'])
    graphs = read(SC/'domain_graphs.csv').set_index(['example','state','scope'])
    jobs = {}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json', EXT/'routing_plan.json', OUT/'routing_plan.json']:
        for case in read(path)['cases']:
            for j in case['jobs']:
                for name in {j['state_id'], *j.get('aliases', [])}:
                    jobs[case['example'], name] = j
    assert all((r.example,r.state) in jobs for r in d.itertuples())
    parents = {ex:read(jobs[ex,'S000_base']['parameters']) for ex in d.example.unique()}
    cache, rows = {}, []
    for ex, group in d.groupby('example'):
        for r in group.itertuples():
            z = dict(example=ex,state=r.state)
            if r.cohort == 'segmented':
                line = read(EXT/f'variants/{ex}/{r.state}/lineage.csv')
                line = line[line.role.eq('collapsed_block')]
            else:
                g = graphs.loc[(ex,r.state,meta.loc[(ex,r.state),'B_scope'])]
            for side, full in [('parent',parents[ex]),('child',read(jobs[ex,r.state]['parameters']))]:
                if r.cohort == 'segmented':
                    ids = set(line[side+'_link_id'])
                    assert ids <= set(full.id_link)
                    f = full[full.id_link.isin(ids)]
                    w = (f.reference_flow_fraction*f.link_length_m).to_numpy()
                else:
                    key = g[side+'_graph']
                    if key not in cache:
                        geom = read(SC/'graphs'/f'{key}.parquet')
                        cache[key] = geom, allocation_tables(geom)[1].set_index('id_link')
                    geom, a = cache[key]
                    f, _ = validate_domain(full,geom)
                    f = f.set_index('id_link').loc[a.index]
                    np.testing.assert_allclose(f[['link_length_m','width']],geom.set_index('id_link').loc[a.index,['link_length_m','width']],rtol=1e-10,atol=1e-7)
                    w = a.mean_length_contribution_m.to_numpy()
                h, s = f.depth_m.to_numpy(), f.slope.to_numpy()
                assert np.isfinite(h).all() and np.isfinite(s).all() and (h>0).all() and (s>0).all()
                assert np.isfinite(w).all() and (w>=0).all()
                np.testing.assert_allclose(w.sum(),getattr(r,'B_'+side+'_route_length_mean'),rtol=1e-10,atol=1e-7)
                # L and h are both metres: dimensionless, with no hidden /1000.
                z[side+'_G_inverse_depth'] = float(np.sum(w/(h*np.sqrt(s))))
                z[side+'_mean_depth_m'] = float(np.average(h,weights=w))
            z[FEATURE] = z['child_G_inverse_depth']-z['parent_G_inverse_depth']
            rows.append(z)
        print(ex,'prepared',len(group),'cases',flush=True)
    d = d.merge(pd.DataFrame(rows),on=['example','state'],how='left',validate='one_to_one')
    assert len(d)==2185 and d.example.nunique()==15 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[COLUMNS+[FEATURE]]).all().all()
    d.to_parquet(DEST/'input.parquet',index=False)
    m = dict(sources=dict(SOURCES),script_hash=sha(__file__),input_hash=sha(DEST/'input.parquet'),rows=len(d),unmatched=0,
        definition='G_h=sum r_e L_e/(h_e sqrt(s_e)), L and h in metres; delta child-parent. Regular B comparison domain; segmented edited footprint excluding connectors.',
        assumptions='Inverse depth and inverse square-root slope. No Manning 2/3 exponent. Stored depth still derived from width^.65 and reference depth; observed representative reach depth could replace it. No roughness, celerity, RAPID K/X or hydrograph-derived feature.',production_modified=False)
    for path,h in m['sources'].items():
        assert sha(path)==h,path
    (DEST/'preparation.json').write_text(json.dumps(m,indent=2)+'\n')
    return d,m


def summarize(new):
    old = read(OUT/'width_joint_resistance_20261023/predictions.parquet')
    old = old[old.panel.isin(['baseline','joint_geometry'])].copy()
    separate = read(OUT/'six_panel_parent_depth_20261023/predictions.parquet')
    separate = separate[separate.depth_added & separate.panel.eq('2_current_plus_Gs')].copy()
    separate['panel'] = 'length_slope_plus_mean_depth'
    cols = ['example','state','cohort','centroid_shift_hours','predicted_hours','panel']
    p = pd.concat([old[cols],separate[cols],new[cols]],ignore_index=True)
    assert len(p)==4*2185 and not p.duplicated(['panel','example','state']).any()
    baseline = old[old.panel.eq('baseline')]
    for _,g in p.groupby('panel'):
        paired = g.merge(baseline,on=['example','state'],validate='one_to_one',suffixes=('_a','_b'))
        assert len(paired)==2185 and (paired.cohort_a==paired.cohort_b).all()
        np.testing.assert_array_equal(paired.centroid_shift_hours_a,paired.centroid_shift_hours_b)
    p.to_parquet(DEST/'predictions.parquet',index=False)
    scores, networks = [], []
    for (panel,cohort),g in p.groupby(['panel','cohort']):
        for subset,z in [('all',g),('without54',g[g.example.ne('54')])]:
            w = 1/z.groupby('example').example.transform('size').to_numpy()
            y = z.centroid_shift_hours.to_numpy()
            err = z.predicted_hours.to_numpy()-y
            scores.append(dict(panel=panel,cohort=cohort,subset=subset,n=len(z),
                r2=1-np.sum(w*err**2)/np.sum(w*(y-np.average(y,weights=w))**2),
                mae_minutes=60*np.average(abs(err),weights=w),
                relative_error_pct=100*np.average(abs(err)/np.maximum(abs(y),1/60),weights=w)))
        for ex,z in g.groupby('example'):
            err=z.predicted_hours-z.centroid_shift_hours
            networks.append(dict(panel=panel,cohort=cohort,example=ex,mae_minutes=60*abs(err).mean(),
                relative_error_pct=100*np.mean(abs(err)/np.maximum(abs(z.centroid_shift_hours),1/60))))
    s=pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv',index=False)
    pd.DataFrame(networks).to_csv(DEST/'network_scores.csv',index=False)
    s[s.subset.eq('all')].groupby('panel').relative_error_pct.mean().rename('equal_cohort_objective_pct').to_csv(DEST/'objective.csv')
    priorities=read(OUT/'outlier_diagnosis_20261023/priority_cases.csv')[['example','state']]
    p.merge(priorities,on=['example','state'],validate='many_to_one').to_csv(DEST/'priority_predictions.csv',index=False)
    print(s.to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',action='store_true')
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args()
    d,prep=prepare()
    choices=[x for x in read(OUT/'nine_input_tuning/selections.json') if x['variant']=='same_grid']
    assert len(choices)==15 and {x['held'] for x in choices}==set(d.example)
    design=dict(preparation=prep,choices=choices,columns=COLUMNS+[FEATURE],seed=b.engine.SEED,
        dependencies={str(Path(m.__file__)):sha(m.__file__) for m in [b,b.engine,b.forest]},
        validation='15 whole-example holdouts, fixed previously selected fold settings; equal-network training weights, training-only preprocessing. Full centroid target, shared 0.5% window. No retuning or augmentation.',fits=15,production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    (DEST/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    if not args.run:return
    rows=[]
    b.COLUMNS=COLUMNS+[FEATURE]
    for choice in choices:
        ex=choice['held'];path=DEST/'folds'/f'{ex}.json';path.parent.mkdir(exist_ok=True)
        if path.exists():
            result=json.loads(path.read_text());assert result['fingerprint']==fp
        else:
            tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)];assert tr.example.nunique()==14
            model=b.fit(tr,choice['params'],jobs=args.jobs);pr=b.forest.predict(model,te);assert np.isfinite(pr).all()
            result=dict(fingerprint=fp,rows=[dict(example=r.example,state=r.state,cohort=r.cohort,
                centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),panel='inverse_depth_joint')
                for r,v in zip(te.itertuples(),pr)])
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
            print(ex,'fit saved',flush=True)
        rows.extend(result['rows'])
    b.COLUMNS=list(COLUMNS)
    summarize(pd.DataFrame(rows))
    (DEST/'completion.json').write_text(json.dumps(dict(complete=True,fingerprint=fp,fits=15,
        matched=2185,unmatched=0,comparison_sources=SOURCES,production_modified=False),indent=2)+'\n')


if __name__=='__main__':main()
