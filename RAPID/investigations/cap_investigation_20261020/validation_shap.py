"""Held-out TreeSHAP of saved nested validation selections; no new tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from alternative_validation import b, OUT, save, sha

SOURCE=OUT/'alternative_validation_20261023'
DEST=OUT/'validation_shap_20261024'
LABELS={
 'log1p__B_ratio_route_length_mean':'ln(1 + μL,c / μL,p)',
 'domain_size':'SL (edited parent size)',
 'log1p__B_ratio_route_length_sd':'ln(1 + σL,c / σL,p)',
 'log1p__B_ratio_width_mean':'ln(1 + W̄c / W̄p)',
 'log1p__B_ratio_width_cv':'ln(1 + CVW,c / CVW,p)',
 'log1p__B_ratio_route_entropy':'ln(1 + Hc / Hp)',
 'B_parent_dominance_mean':'d̄p',
 'parent_slope_slow':'Parent slope factor',
 'length_delta_km':'ΔμL / km',
 'resistance_explicit_depth':'−log10(Keq,c / Keq,p)',
 'delta_G_inverse_depth':'ΔGh',
}


def prepare():
    import shap, sklearn
    d=pd.read_parquet(SOURCE/'input.parquet')
    design=json.loads((SOURCE/'design.json').read_text())
    splits=json.loads((SOURCE/'splits.json').read_text())['tasks']
    selections=json.loads((SOURCE/'selections.json').read_text())
    tasks=[]
    for s in selections:
        t=next(t for t in splits if all(t[k]==s[k] for k in ['strategy','repeat','fold']))
        tasks.append({k:s[k] for k in ['strategy','repeat','fold','model','params']} | dict(train=t['train_indices'],test=t['test_indices']))
    files=[SOURCE/x for x in ['input.parquet','design.json','splits.json','selections.json','predictions.parquet']]
    for model,folder,variant in [('geometry_9','nine_input_tuning','same_grid'),('depth_informed_10','ten_input_tuning_20261023','nested_tuned')]:
        path=OUT/folder/'selections.json';files.extend([path,OUT/folder/'predictions.parquet'])
        for s in json.loads(path.read_text()):
            if 'variant' in s and s['variant']!=variant:continue
            mask=d.example.eq(s['held']).to_numpy()
            tasks.append(dict(strategy='leave_one_network_out',repeat=0,fold=s['held'],model=model,params=s['params'],train=np.flatnonzero(~mask).tolist(),test=np.flatnonzero(mask).tolist()))
    files.extend([Path(__file__),Path(b.__file__),Path(b.engine.__file__),Path(b.forest.__file__)])
    manifest=dict(sources={str(f):sha(f) for f in files},panels=design['panels'],fits=len(tasks),
        shap_version=shap.__version__,sklearn_version=sklearn.__version__,
        method='Exact tree_path_dependent TreeSHAP, training tree cover as reference; no held-out background. Explain all held-out rows. Refit saved selected outer forests, verify predictions against saved OOF values. Target-standardization reversed to minutes.',
        aggregation='Within each cohort and repeat: equal network weight. Mean absolute contributions before aggregation across repeats. Repeat ranges are descriptive, not confidence intervals. No averaging signed SHAP before absolute importance.',
        limitations='Explains model predictions, not causality or error contributions. Correlated predictors share credit under this dependence convention. Fold references differ; no common-reference causal interpretation. Geometry LOEO uses original four-candidate grid, new strategies six.',
        production_modified=False)
    fp=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if (DEST/'manifest.json').exists():assert json.loads((DEST/'manifest.json').read_text())==manifest
    save(DEST/'manifest.json',manifest)
    return tasks,design['panels'],fp


def run_task(task,cols,fp):
    import shap
    key=f"{task['strategy']}/r{task['repeat']}/f{task['fold']}/{task['model']}"
    path=DEST/'checkpoints'/key/'shap.parquet';meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        assert json.loads(meta.read_text())['fingerprint']==fp
        return key
    d=pd.read_parquet(SOURCE/'input.parquet');tr=d.iloc[task['train']];te=d.iloc[task['test']]
    assert set(tr.index).isdisjoint(te.index)
    if task['strategy']=='compound_mixed':assert set(tr.group_id).isdisjoint(te.group_id)
    else:assert set(tr.example).isdisjoint(te.example)
    b.COLUMNS=list(cols);a=b.fit(tr,task['params'],jobs=1)
    pred=b.forest.predict(a,te)
    if task['strategy']=='leave_one_network_out':
        folder,variant=('nine_input_tuning','same_grid') if task['model']=='geometry_9' else ('ten_input_tuning_20261023','nested_tuned')
        old=pd.read_parquet(OUT/folder/'predictions.parquet');old=old[old.variant.eq(variant)]
    else:
        old=pd.read_parquet(SOURCE/'predictions.parquet')
        old=old[(old.strategy==task['strategy'])&(old.repeat==task['repeat'])&(old.model==task['model'])]
    joined=te[['example','state']].merge(old[['example','state','predicted_hours']],on=['example','state'],validate='one_to_one',how='left')
    assert joined.predicted_hours.notna().all()
    np.testing.assert_allclose(pred,joined.predicted_hours,rtol=1e-10,atol=1e-10)
    pre=a['preprocessor'];x=pre.transform(te)
    explainer=shap.TreeExplainer(a['estimator'],feature_perturbation='tree_path_dependent',model_output='raw')
    sv=np.asarray(explainer.shap_values(x,check_additivity=True))*a['target_sd']*60
    base=float(np.asarray(explainer.expected_value).item())*a['target_sd']*60+a['target_mean']*60
    np.testing.assert_allclose(base+sv.sum(axis=1),pred*60,rtol=1e-8,atol=1e-7)
    out=te[['example','state','cohort','centroid_shift_hours']].reset_index(drop=True).copy()
    for k in ['strategy','repeat','fold','model']:out[k]=task[k]
    out['predicted_minutes']=pred*60;out['base_minutes']=base
    for c in cols:
        out['value__'+c]=te[c].to_numpy()
        out['shap__'+c]=sv[:,pre.kept.index(c)] if c in pre.kept else 0.
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');out.to_parquet(tmp,index=False);tmp.replace(path)
    save(meta,dict(fingerprint=fp,dropped=pre.dropped,rows=len(te),prediction_max_error_hours=float(np.max(abs(pred-joined.predicted_hours))),additivity_max_error_minutes=float(np.max(abs(base+sv.sum(axis=1)-pred*60)))))
    return key


def summarize(panels):
    rows=pd.concat([pd.read_parquet(p) for p in sorted((DEST/'checkpoints').glob('*/*/*/*/shap.parquet'))],ignore_index=True)
    assert not rows.duplicated(['strategy','repeat','model','example','state']).any()
    assert rows.groupby(['strategy','repeat','model']).size().eq(2185).all()
    assert len(rows)==48070
    rows['fold']=rows['fold'].astype(str)
    rows.to_parquet(DEST/'values.parquet',index=False)
    metrics=[]
    for (strategy,repeat,model,cohort),g in rows.groupby(['strategy','repeat','model','cohort']):
        w=1/g.groupby('example').example.transform('size').to_numpy(float);w/=w.sum()
        vals=[]
        for c in panels[model]:
            v=g['shap__'+c].to_numpy();x=g['value__'+c].to_numpy()
            # Equal-network rank association, descriptive direction only.
            xr=pd.Series(x).rank().to_numpy();vr=pd.Series(v).rank().to_numpy()
            xc=xr-np.average(xr,weights=w);vc=vr-np.average(vr,weights=w)
            denom=np.sqrt(np.sum(w*xc*xc)*np.sum(w*vc*vc))
            vals.append(dict(feature=c,label=LABELS[c],mean_abs_minutes=np.sum(w*abs(v)),mean_signed_minutes=np.sum(w*v),rank_association=np.sum(w*xc*vc)/denom if denom else np.nan))
        total=sum(v['mean_abs_minutes'] for v in vals)
        for v in vals:metrics.append(dict(strategy=strategy,repeat=repeat,model=model,cohort=cohort,share_pct=100*v['mean_abs_minutes']/total,**v))
    m=pd.DataFrame(metrics);m['rank']=m.groupby(['strategy','repeat','model','cohort']).mean_abs_minutes.rank(ascending=False,method='min')
    m.to_csv(DEST/'importance_by_repeat.csv',index=False)
    s=m.groupby(['strategy','model','cohort','feature','label'])[['mean_abs_minutes','share_pct','rank','rank_association']].agg(['mean','min','max'])
    s.columns=['_'.join(c) for c in s.columns];s=s.reset_index();s.to_csv(DEST/'importance_summary.csv',index=False)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    pieces=['<html><head><meta charset="utf-8"></head><body><h1>Held-out SHAP across validation strategies</h1><p>Contributions are minutes relative to each fitted forest reference. Negative: earlier centroid; positive: later. Model explanations, not causal effects. Equal network weights for importance; separate cohorts. Scatter points are individual states (not weighted density), repeat 0 for clarity; bars include all five repeats.</p>']
    first=True
    for model in panels:
        fig=make_subplots(rows=1,cols=2,subplot_titles=['Regular units','Segmented full collapses'])
        for ci,cohort in enumerate(['original','segmented'],1):
            z=s[(s.model==model)&(s.cohort==cohort)]
            order=z.groupby('label').mean_abs_minutes_mean.mean().sort_values().index.tolist()
            for strategy,g in z.groupby('strategy'):
                g=g.set_index('label').reindex(order)
                fig.add_trace(go.Bar(name=strategy,x=g.mean_abs_minutes_mean,y=order,orientation='h',legendgroup=strategy,showlegend=ci==1,
                    error_x=dict(type='data',symmetric=False,array=g.mean_abs_minutes_max-g.mean_abs_minutes_mean,arrayminus=g.mean_abs_minutes_mean-g.mean_abs_minutes_min)),row=1,col=ci)
        fig.update_layout(title=model,height=620,width=1300,barmode='group',margin=dict(l=220));fig.update_xaxes(title_text='Mean |SHAP| (minutes), equal network weights')
        pieces.append(fig.to_html(full_html=False,include_plotlyjs=True if first else False));first=False
        for cohort in ['original','segmented']:
            fig=make_subplots(rows=len(panels[model]),cols=3,subplot_titles=[f'{LABELS[c]} — {st}' for c in panels[model] for st in ['leave_one_network_out','network_balanced','compound_mixed']],vertical_spacing=.025)
            for ri,c in enumerate(panels[model],1):
                for ci,st in enumerate(['leave_one_network_out','network_balanced','compound_mixed'],1):
                    z=rows[(rows.model==model)&(rows.cohort==cohort)&(rows.strategy==st)&(rows.repeat==0)]
                    fig.add_trace(go.Scattergl(x=z['value__'+c],y=z['shap__'+c],mode='markers',marker=dict(size=5,opacity=.5),text=z.example+' / '+z.state,hovertemplate='%{text}<br>Input=%{x}<br>SHAP=%{y:.2f} min<extra></extra>',showlegend=False),row=ri,col=ci)
                    fig.add_hline(y=0,line_width=.5,row=ri,col=ci)
            fig.update_layout(title=f'{model}: {cohort} — held-out dependence, repeat 0',height=260*len(panels[model]),width=1400)
            pieces.append(fig.to_html(full_html=False,include_plotlyjs=False))
    pieces.append('</body></html>');(DEST/'report.html').write_text('\n'.join(pieces))
    save(DEST/'completion.json',dict(complete=True,rows=len(rows),fits=130,production_modified=False))
    print(s.sort_values('rank_mean').groupby(['model','cohort','strategy']).head(4)[['model','cohort','strategy','label','share_pct_mean','rank_mean']].to_string(index=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    tasks,panels,fp=prepare();print(f'{len(tasks)} saved outer fits; no tuning.',flush=True)
    if not args.run:return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(run_task,t,panels[t['model']],fp) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):print(i,'/130',f.result(),flush=True)
    summarize(panels)

if __name__=='__main__':main()
