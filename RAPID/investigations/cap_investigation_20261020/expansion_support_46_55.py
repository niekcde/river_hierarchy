"""Held-out error and training support audit for segmented 46/55; no new fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'RAPID/outputs/uncapped_event_window_20261020/combination_grouped_validation_20261026'
DEST=SOURCE/'support_46_55'


def main():
    DEST.mkdir(exist_ok=True)
    d=pd.read_parquet(SOURCE/'input.parquet')
    p=pd.read_parquet(SOURCE/'predictions.parquet')
    tasks=json.loads((SOURCE/'tasks.json').read_text())['tasks']
    rows=[];nearest=[]
    for ex in ['46','55']:
        ids=np.flatnonzero(d.example.eq(ex)&d.cohort.eq('segmented'));assert len(ids)==1
        idx=int(ids[0]);r=d.iloc[idx];y=float(r.centroid_shift_hours)
        for task in tasks:
            if idx not in task['test']:continue
            base=next(t for t in tasks if t['variant']=='baseline' and t['repeat']==task['repeat'] and t['fold']==task['fold'] and t['model']==task['model'])
            tr=d.iloc[task['train']].copy();b=d.iloc[base['train']];cols=task['columns']
            assert r.group_id not in set(tr.group_id)
            # Fixed baseline-training scales within each matched fold: added
            # rows cannot look closer merely by widening the normalizer.
            scale=b[cols].quantile(.75)-b[cols].quantile(.25)
            scale=scale.where(scale>1e-12,b[cols].std()).fillna(1).replace(0,1)
            tr['distance']=np.sqrt((((tr[cols]-r[cols].astype(float))/scale)**2).mean(axis=1))
            other=tr[tr.example.ne(ex)]
            comparable=tr[(np.sign(tr.centroid_shift_hours)==np.sign(y)) & tr.centroid_shift_hours.abs().between(abs(y)/2,abs(y)*2)]
            pred=p[(p.example==ex)&p.cohort.eq('segmented')&(p['repeat']==task['repeat'])&(p.model==task['model'])&(p.variant==task['variant'])]
            assert len(pred)==1
            outside=[c for c in cols if r[c]<tr[c].min() or r[c]>tr[c].max()]
            low,high=tr.centroid_shift_hours.min(),tr.centroid_shift_hours.max()
            weights=1/tr.groupby('example').example.transform('size')
            row=dict(example=ex,model=task['model'],variant=task['variant'],repeat=task['repeat'],fold=task['fold'],
                target_hours=y,prediction_hours=float(pred.predicted_hours.iloc[0]),
                error_hours=float(abs(pred.predicted_hours.iloc[0]-y)),train_target_min=low,train_target_max=high,
                target_range_error_floor_hours=max(low-y,y-high,0),outside_features=' | '.join(outside),outside_feature_count=len(outside),
                nearest_distance=float(tr.distance.min()),nearest_other_network_distance=float(other.distance.min()),
                nearest5_mean_distance=float(tr.nsmallest(5,'distance').distance.mean()),
                comparable_rows=len(comparable),comparable_networks=comparable.example.nunique(),
                comparable_same_network_rows=int(comparable.example.eq(ex).sum()),
                same_network_combinations=int((tr.example.eq(ex)&tr.cohort.eq('combined')).sum()),
                comparable_training_weight_pct=float(100*weights.loc[comparable.index].sum()/weights.sum()))
            rows.append(row)
            for rank,n in enumerate(tr.sort_values(['distance','example','state']).head(5).itertuples(),1):
                nearest.append(dict(example=ex,model=task['model'],variant=task['variant'],repeat=task['repeat'],rank=rank,
                    neighbor_example=n.example,neighbor_state=n.state,neighbor_cohort=n.cohort,distance=n.distance,target_hours=n.centroid_shift_hours))
    result=pd.DataFrame(rows);assert len(result)==60
    result.to_csv(DEST/'fold_support.csv',index=False)
    pd.DataFrame(nearest).to_csv(DEST/'nearest_training_cases.csv',index=False)
    metrics=['error_hours','prediction_hours','train_target_min','target_range_error_floor_hours','outside_feature_count',
        'nearest_distance','nearest_other_network_distance','nearest5_mean_distance','comparable_rows','comparable_networks',
        'comparable_same_network_rows','same_network_combinations','comparable_training_weight_pct']
    summary=result.groupby(['example','model','variant'])[metrics].agg(['mean','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(DEST/'summary.csv',index=False)
    before=result[result.variant.eq('baseline')]
    changes=result[result.variant.ne('baseline')].merge(before,on=['example','model','repeat'],suffixes=('','_baseline'),validate='many_to_one')
    changes['error_change_hours']=changes.error_hours-changes.error_hours_baseline
    changes.to_csv(DEST/'paired_changes.csv',index=False)
    files=[SOURCE/n for n in ['input.parquet','predictions.parquet','tasks.json']]+[Path(__file__)]
    (DEST/'manifest.json').write_text(json.dumps(dict(sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        fits=0,held_out_cases=2,rows=60,method='RMS predictor distance using fixed baseline-training IQR per matched outer fold; fallback training SD then1. Same-sign factor-two target band is descriptive only. Compare identical held-out cases and folds.',
        limitations='Known-network singleton exception permits constituent overlap. Distance is descriptive, not forest leaf similarity. No controlled attribution between same-network and other-network additions; no new independent networks.'),indent=2)+'\n')
    print(result.groupby(['example','model','variant'])[metrics].mean().round(3).to_string())
    print('\nImproved repetitions:')
    print(changes.assign(improved=changes.error_change_hours.lt(0)).groupby(['example','model','variant']).improved.sum().to_string())


if __name__=='__main__':main()
