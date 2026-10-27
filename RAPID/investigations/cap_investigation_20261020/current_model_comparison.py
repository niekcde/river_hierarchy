"""Four cohort scores, final RF settings, and current tuned-model outliers."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from slope_ratio_audit import OUT
from ten_input_outlier_audit import score,weights

DEST=OUT/'current_model_comparison_20261023'


def main():
    DEST.mkdir(exist_ok=True)
    nine=OUT/'nine_input_tuning';ten=OUT/'ten_input_tuning_20261023'
    a=pd.read_parquet(nine/'predictions.parquet');a=a[a.variant.eq('same_grid')].copy()
    b=pd.read_parquet(ten/'predictions.parquet');b=b[b.variant.eq('nested_tuned')].copy()
    check=a.merge(b,on=['example','state'],validate='one_to_one',suffixes=('_a','_b'))
    assert len(a)==len(b)==len(check)==2185 and (check.cohort_a==check.cohort_b).all()
    np.testing.assert_array_equal(check.centroid_shift_hours_a,check.centroid_shift_hours_b)
    a['model']='geometry_9';b['model']='depth_informed_10'
    s=[]
    for model,d in [('geometry_9',a),('depth_informed_10',b)]:
        for cohort,g in d.groupby('cohort'):
            s.append(dict(model=model,cohort=cohort,**score(g,'equal_network')))
    scores=pd.DataFrame(s);scores.to_csv(DEST/'four_scores.csv',index=False)
    settings={}
    for label,path in [('geometry_9',nine/'research_model_same_grid.json'),('depth_informed_10',ten/'final_model.json')]:
        m=json.loads(path.read_text())
        modelpath=path.with_suffix('.joblib')
        assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==m['sha256']
        settings[label]=dict(params=m['selected']['params'],columns=m['columns'],artifact=str(modelpath),
            seed=json.loads((path.parent/'design.json').read_text())['seed'])
    (DEST/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    parts=[]
    for cohort,g in b.groupby('cohort'):
        g=g.copy();e=g.predicted_hours-g.centroid_shift_hours;w=weights(g,'equal_network')
        g['abs_error_hours']=abs(e)
        g['adjusted_relative_error_pct']=100*abs(e)/np.maximum(abs(g.centroid_shift_hours),1/60)
        g['cohort_sse_share_pct']=100*w*e**2/np.sum(w*e**2)
        g['cohort_relative_share_pct']=100*w*g.adjusted_relative_error_pct/np.sum(w*g.adjusted_relative_error_pct)
        g['sse_rank']=g.cohort_sse_share_pct.rank(ascending=False,method='first').astype(int)
        parts.append(g)
    ranked=pd.concat(parts).sort_values(['cohort','sse_rank'])
    ranked.to_parquet(DEST/'ranked_tuned_outliers.parquet',index=False)
    top=ranked.groupby('cohort').head(5)
    top.to_csv(DEST/'largest_tuned_outliers.csv',index=False)
    ranked.sort_values('cohort_relative_share_pct',ascending=False).groupby('cohort').head(5).to_csv(DEST/'largest_relative_contributors.csv',index=False)
    paths=[nine/'predictions.parquet',nine/'research_model_same_grid.json',nine/'design.json',
        ten/'predictions.parquet',ten/'final_model.json',ten/'design.json',Path(__file__),Path(__file__).with_name('ten_input_outlier_audit.py')]
    (DEST/'manifest.json').write_text(json.dumps(dict(sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        matched=2185,unmatched=0,ranking='Within-cohort equal-network weighted squared-error contributions.',
        comparison='Accepted nine-input same_grid procedure (four candidates) versus ten-input nested tuning (six candidates). Both outer leave-one-network-out, inner three GroupKFold folds. All-data model settings differ from some outer-fold settings.',
        validation='No response-stratified fold assignment used in these results. No new fits or production changes.'),indent=2)+'\n')
    print(scores.to_string(index=False))
    print(top[['example','state','cohort','centroid_shift_hours','predicted_hours','abs_error_hours','cohort_sse_share_pct']].to_string(index=False))


if __name__=='__main__':main()
