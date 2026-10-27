"""Six fresh fixed-settings panels: current, resistance, inverse-depth delta G."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import six_panel_geometry_depth_resistance as resistance
import inverse_depth_joint_check as joint

b = resistance.b
DEST = resistance.OUT/'resistance_inverse_depth_panels_20261023'
BASE = list(resistance.COLUMNS)
R, G = resistance.R, joint.FEATURE
REPLACED = [R if c == resistance.WIDTH else c for c in BASE]
PANELS = {
    '1_current': BASE,
    '2_current_plus_Gh': BASE+[G],
    '3_replace_width_with_Rh': REPLACED,
    '4_current_plus_Rh': BASE+[R],
    '5_replace_width_with_Rh_plus_Gh': REPLACED+[G],
    '6_current_plus_Rh_plus_Gh': BASE+[R,G],
}


def summarize(p):
    scores, networks = [], []
    for (panel, cohort), group in p.groupby(['panel','cohort']):
        for subset,z in [('all',group),('without54',group[group.example.ne('54')])]:
            w=1/z.groupby('example').example.transform('size').to_numpy()
            y=z.centroid_shift_hours.to_numpy()
            err=z.predicted_hours.to_numpy()-y
            scores.append(dict(panel=panel,cohort=cohort,subset=subset,n=len(z),
                r2=1-np.sum(w*err**2)/np.sum(w*(y-np.average(y,weights=w))**2),
                mae_minutes=60*np.average(abs(err),weights=w),
                relative_error_pct=100*np.average(abs(err)/np.maximum(abs(y),1/60),weights=w)))
        for ex,z in group.groupby('example'):
            err=z.predicted_hours-z.centroid_shift_hours
            networks.append(dict(panel=panel,cohort=cohort,example=ex,
                mae_minutes=60*abs(err).mean(),
                relative_error_pct=100*np.mean(abs(err)/np.maximum(abs(z.centroid_shift_hours),1/60))))
    s=pd.DataFrame(scores)
    s.to_csv(DEST/'scores.csv',index=False)
    pd.DataFrame(networks).to_csv(DEST/'network_scores.csv',index=False)
    objective=s.groupby(['subset','panel']).relative_error_pct.mean().rename('equal_cohort_objective_pct')
    objective.to_csv(DEST/'objective.csv')
    print(s[s.subset.eq('all')].to_string(index=False),flush=True)
    print(objective.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    paths=[Path(__file__)]
    for folder in [resistance.DEST,joint.DEST]:
        paths += [folder/x for x in ['input.parquet','preparation.json','design.json','completion.json','predictions.parquet']]
        assert json.loads((folder/'completion.json').read_text())['complete']
    paths += [Path(m.__file__) for m in [resistance,joint,b,b.engine,b.forest]]
    priority_path=resistance.OUT/'outlier_diagnosis_20261023/priority_cases.csv'
    paths.append(priority_path)
    hashes={str(p):joint.sha(p) for p in paths}
    rd=json.loads((resistance.DEST/'design.json').read_text())
    gd=json.loads((joint.DEST/'design.json').read_text())
    assert rd['choices']==gd['choices'] and rd['seed']==gd['seed']==b.engine.SEED
    for path,h in rd['dependencies'].items():
        assert joint.sha(path)==h,path
    for path,h in gd['dependencies'].items():
        assert joint.sha(path)==h,path
    # Both descriptor preparations must refer to identical shared inputs.
    rp=json.loads((resistance.DEST/'preparation.json').read_text())
    gp=json.loads((joint.DEST/'preparation.json').read_text())
    for path in rp['sources'].keys() & gp['sources'].keys():
        assert rp['sources'][path]==gp['sources'][path],path
    assert gp['input_hash']==hashes[str(joint.DEST/'input.parquet')]
    d=pd.read_parquet(joint.DEST/'input.parquet')
    r=pd.read_parquet(resistance.DEST/'input.parquet')
    check=d.merge(r,on=['example','state'],validate='one_to_one',suffixes=('_g','_r'))
    assert len(check)==len(d)==len(r)==2185
    for col in BASE+['centroid_shift_hours']:
        np.testing.assert_array_equal(check[col+'_g'],check[col+'_r'])
    assert (check.cohort_g==check.cohort_r).all()
    d=d.merge(r[['example','state',R]],on=['example','state'],validate='one_to_one',how='left')
    assert d.example.nunique()==15 and np.isfinite(d[BASE+[R,G]]).all().all()
    d.to_parquet(DEST/'input.parquet',index=False)
    choices=rd['choices']
    assert len(choices)==15 and {x['held'] for x in choices}==set(d.example)
    design=dict(sources=hashes,panels=PANELS,choices=choices,seed=b.engine.SEED,fits=90,
        input_hash=joint.sha(DEST/'input.parquet'),
        validation='Fresh fits for every panel. Same prior fold-selected settings; 15 whole-example holdouts, equal-network training weights, training-only preprocessing. Full centroid target, 0.5% shared window; no retuning or augmentation.',
        G='Delta sum r_e L_e/(h_e sqrt(s_e)); L,h metres. Regular B domain; segmented edited footprint excluding connectors. No separate mean-depth predictor.',
        R='-log10(K_child/K_parent), edge K=L/(W^2 h^3). Regular B domain; segmented full N including connectors. Existing explicit-depth resistance unchanged, not RAPID travel-time K.',
        caveat='Both candidates currently use derived depth. The requested removal of Manning 2/3 applies to G, not to the established resistance law. Reused exploratory folds are not an independent final test.',production_modified=False)
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    if (DEST/'design.json').exists():
        assert json.loads((DEST/'design.json').read_text())==design,'Existing design differs'
    (DEST/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    if not args.run:
        print('Prepared 90 fits; --run to execute.');return
    rows=[]
    for panel,cols in PANELS.items():
        b.COLUMNS=list(cols)
        for choice in choices:
            ex=choice['held'];path=DEST/'folds'/panel/f'{ex}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                result=json.loads(path.read_text());assert result['fingerprint']==fp
            else:
                tr=d[d.example.ne(ex)];te=d[d.example.eq(ex)];assert tr.example.nunique()==14
                model=b.fit(tr,choice['params'],jobs=args.jobs)
                pred=b.forest.predict(model,te);assert np.isfinite(pred).all()
                result=dict(fingerprint=fp,rows=[dict(example=r.example,state=r.state,cohort=r.cohort,
                    centroid_shift_hours=r.centroid_shift_hours,predicted_hours=float(v),panel=panel)
                    for r,v in zip(te.itertuples(),pred)])
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
                print(panel,ex,'saved',flush=True)
            rows.extend(result['rows'])
    b.COLUMNS=list(BASE)
    p=pd.DataFrame(rows)
    assert len(p)==6*2185 and not p.duplicated(['panel','example','state']).any()
    # Repeated panels must reproduce earlier predictions within floating-point tolerance.
    oldr=pd.read_parquet(resistance.DEST/'predictions.parquet')
    oldg=pd.read_parquet(joint.DEST/'predictions.parquet')
    for panel,old,oldpanel in [('1_current',oldr,'1_current'),
            ('2_current_plus_Gh',oldg,'inverse_depth_joint'),
            ('3_replace_width_with_Rh',oldr,'3_replace_width_with_Rh'),
            ('4_current_plus_Rh',oldr,'4_current_plus_Rh')]:
        a=p[p.panel.eq(panel)].merge(old[old.panel.eq(oldpanel)],on=['example','state'],validate='one_to_one',suffixes=('_new','_old'))
        assert len(a)==2185
        np.testing.assert_array_equal(a.centroid_shift_hours_new,a.centroid_shift_hours_old)
        np.testing.assert_allclose(a.predicted_hours_new,a.predicted_hours_old,rtol=1e-12,atol=1e-12)
    p.to_parquet(DEST/'predictions.parquet',index=False)
    summarize(p)
    priority=pd.read_csv(priority_path,dtype={'example':str})[['example','state']]
    p.merge(priority,on=['example','state'],validate='many_to_one').to_csv(DEST/'priority_predictions.csv',index=False)
    for path,h in hashes.items():assert joint.sha(path)==h,path
    (DEST/'completion.json').write_text(json.dumps(dict(complete=True,fingerprint=fp,fits=90,
        matched_per_panel=2185,unmatched=0,reproduced_prior_panels=4,production_modified=False),indent=2)+'\n')


if __name__=='__main__':main()
