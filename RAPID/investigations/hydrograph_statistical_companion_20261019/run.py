"""Reproducible statistical companion; see DESIGN.md for the research contract."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'RAPID/outputs/hydrograph_statistical_companion_20261019'
BASE=ROOT/'RAPID/outputs/hydrograph_no_exposure_20261019'
INPUT=ROOT/'RAPID/outputs/hydrograph_relative_sweep_20261019/input.parquet'
COLS=json.loads((BASE/'final_model.json').read_text())['columns']
TERMS=['intercept']+COLS
NETWORKS='02 03 05 13 15 21 22 37 41 45 46 48 54 55 56'.split()
SEED=20261019

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj): p.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n')
def weights(d): return 1/d.groupby('example').example.transform('size').to_numpy(float)
def score(d,p):
    y=d.centroid_shift_hours.to_numpy()*60; p=np.asarray(p)*60;w=weights(d)
    var=np.average((y-np.average(y,weights=w))**2,weights=w)
    return dict(n=len(d),networks=d.example.nunique(),r2=1-np.average((y-p)**2,weights=w)/var if var>0 else np.nan,mae_minutes=np.average(abs(y-p),weights=w),relative_error_pct=100*np.average(abs(y-p)/np.maximum(abs(y),1),weights=w))

def fit(d,model):
    raw=d[COLS].to_numpy(float); y=d.centroid_shift_hours.to_numpy(float)
    w=weights(d); mu=np.average(raw,axis=0,weights=w);sd=np.sqrt(np.average((raw-mu)**2,axis=0,weights=w))
    if np.any(sd<1e-12): raise ValueError('Constant predictor')
    x=np.column_stack([np.ones(len(d)),(raw-mu)/sd]);n,k=x.shape
    if np.linalg.matrix_rank(x)<k:raise ValueError('Rank-deficient design')
    info={}
    if model=='wls':
        b=np.linalg.lstsq(x*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    else:
        codes,groups=pd.factorize(d.example); counts=np.bincount(codes).astype(float)
        sx=np.zeros((len(groups),k));sy=np.zeros(len(groups));np.add.at(sx,codes,x);np.add.at(sy,codes,y)
        xx=x.T@x;xy=x.T@y;yy=y@y
        def profile(lam,ret=False):
            a=lam/(1+lam*counts); h=xx-sx.T@(a[:,None]*sx);rhs=xy-sx.T@(a*sy)
            sign,ld=np.linalg.slogdet(h)
            if sign<=0:return np.inf
            beta=np.linalg.solve(h,rhs);rss=yy-np.sum(a*sy*sy)-rhs@beta
            if rss<=0:return np.inf
            objective=(n-k)*np.log(rss/(n-k))+np.log1p(lam*counts).sum()+ld
            return (beta,rss/(n-k),objective) if ret else objective
        grid=np.linspace(-20,20,61); vals=np.array([profile(np.exp(t)) for t in grid]);j=int(np.argmin(vals))
        opt=minimize_scalar(lambda z:profile(np.exp(z)),bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded',options={'xatol':1e-9})
        if not opt.success or j==len(grid)-1:raise ValueError('REML search failed or upper bound reached')
        lam=float(np.exp(opt.x)) if opt.fun<profile(0) else 0.
        b,sigma2,obj=profile(lam,True)
        info=dict(lambda_ratio=lam,residual_variance_hours2=sigma2,network_variance_hours2=lam*sigma2,icc=lam/(1+lam),boundary=lam<1e-8,objective=obj)
    native=np.r_[b[0]-np.sum(b[1:]*mu/sd),b[1:]/sd]
    return native,info

def predict(d,b): return b[0]+d[COLS].to_numpy(float)@b[1:]
def mask(d,scenario):
    ok=np.ones(len(d),bool)
    if scenario in ('without54','without_both'):ok &= d.example.ne('54')
    if scenario in ('without_extreme','without_both'):ok &= ~(d.example.eq('37')&d.state.eq('CB001_bubble_3'))
    return ok

def quantile(x,w,p):
    order=np.argsort(x); x=np.asarray(x)[order];w=np.asarray(w)[order]
    return float(np.interp(p,(np.cumsum(w)-.5*w)/w.sum(),x))

def audit(d,p):
    assert len(d)==2170 and sorted(d.example.unique())==NETWORKS
    assert not d.duplicated(['example','state']).any() and not d.state.eq('fully_collapsed').any()
    assert d.window.eq('0.5%').all() and np.isfinite(d[COLS+['centroid_shift_hours']]).all().all()
    joined=d[['example','state','centroid_shift_hours']].merge(p,on=['example','state'],validate='one_to_one',how='outer',indicator=True,suffixes=('_input',''))
    assert joined._merge.eq('both').all();np.testing.assert_array_equal(joined.centroid_shift_hours,joined.centroid_shift_hours_input)
    checks=[]
    for c in COLS:
        expected=np.log1p(d[c.replace('log1p__','')]) if c.startswith('log1p__') else d.B_parent_route_length_mean/1000 if c=='domain_size' else d[c]
        np.testing.assert_array_equal(d[c],expected);checks.append(dict(feature=c,maximum_transform_error=float(np.max(abs(d[c]-expected)))))
    pd.DataFrame(checks).to_csv(OUT/'transform_audit.csv',index=False)
    signature=[c for c in d if c.startswith('B_parent_')]
    rows=[]
    for ex,g in d.groupby('example'):
        assert g.parent_centroid_seconds.nunique()==1
        rows.append(dict(example=ex,n=len(g),response_parent='S000_base',parent_signatures=len(g[signature].drop_duplicates()),overlap_groups=g.overlap_group.nunique(),repeated_parent_rows=g.duplicated(signature,keep=False).sum()))
    pd.DataFrame(rows).to_csv(OUT/'dependence_audit.csv',index=False)
    response=pd.read_parquet(ROOT/'RAPID/outputs/hydrograph_response/responses.parquet')
    match=d[['example','state','centroid_shift_hours']].merge(response[['example','state','centroid_shift_hours']],on=['example','state'],validate='one_to_one',how='left',suffixes=('','_response'))
    np.testing.assert_allclose(match.centroid_shift_hours,match.centroid_shift_hours_response,rtol=0,atol=1e-12)
    coverage=response[['example','state']].merge(d[['example','state']],on=['example','state'],how='outer',indicator=True,validate='one_to_one')
    coverage.to_csv(OUT/'response_coverage.csv',index=False)
    diag=[];w=weights(d)
    for view in ['pooled','within_network']:
        z=d[COLS].copy()
        if view=='within_network':z-=d.groupby('example')[COLS].transform('mean')
        raw=z.to_numpy();m=np.average(raw,axis=0,weights=w);s=np.sqrt(np.average((raw-m)**2,axis=0,weights=w));x=(raw-m)/s
        corr=x.T@(w[:,None]*x)/w.sum();pd.DataFrame(corr,index=COLS,columns=COLS).to_csv(OUT/f'{view}_correlation.csv')
        inv=np.linalg.inv(corr)
        for i,c in enumerate(COLS):diag.append(dict(view=view,feature=c,vif=inv[i,i],condition_number=np.linalg.cond(x*np.sqrt(w[:,None]))))
    pd.DataFrame(diag).to_csv(OUT/'collinearity.csv',index=False)
    baseline=score(joined,joined.predicted_hours);np.testing.assert_allclose([baseline['r2'],baseline['mae_minutes'],baseline['relative_error_pct']],[.6145313418707585,2.822440993149743,63.737195046810854],rtol=1e-10)
    return baseline

def main(boot):
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'RAPID/outputs/hydrograph_response/responses.parquet',INPUT,BASE/'predictions.parquet',BASE/'final_model.json',BASE/'final_model.joblib',ROOT/'RAPID/src/rapid_tools/hydrograph_response.py',Path(__file__),HERE/'DESIGN.md',HERE/'validate.py',HERE/'report.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    d=pd.read_parquet(INPUT);p=pd.read_parquet(BASE/'predictions.parquet');p=p[p.variant.eq('seven_inputs')].copy();baseline=audit(d,p)
    dump(OUT/'manifest.json',dict(status='running',sources=hashes,seed=SEED,bootstrap_draws=boot,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),baseline=baseline))
    coefs=[];preds=[];fits=[];full={}
    for scenario in ['all','without54','without_extreme','without_both']:
        cohort=d[mask(d,scenario)].copy()
        for model in ['mixed','wls']:
            for held in ['full']+sorted(cohort.example.unique()):
                train=cohort if held=='full' else cohort[cohort.example.ne(held)]
                b,info=fit(train,model);fits.append(dict(scenario=scenario,model=model,held=held,**info))
                coefs.extend(dict(scenario=scenario,model=model,held=held,term=t,coefficient_hours=v) for t,v in zip(TERMS,b))
                if held=='full':
                    if scenario=='all':full[model]=b
                else:
                    test=cohort[cohort.example.eq(held)];q=test[['example','state','centroid_shift_hours','peak_shift_hours']].copy();q['predicted_hours']=predict(test,b);q['scenario']=scenario;q['model']=model;preds.append(q)
            print(scenario,model,'complete',flush=True)
    coef=pd.DataFrame(coefs);coef.to_csv(OUT/'coefficients.csv',index=False);pd.DataFrame(fits).to_csv(OUT/'fit_diagnostics.csv',index=False)
    pred=pd.concat(preds,ignore_index=True);pred.to_parquet(OUT/'predictions.parquet',index=False)
    performance=[];pernet=[]
    forest=p[['example','state','centroid_shift_hours','peak_shift_hours','predicted_hours']].copy();forest['model']='frozen_forest';forest['scenario']='all'
    for (model,scenario),g in pd.concat([pred,forest]).groupby(['model','scenario']):
        for view in (['all','without54','without_extreme','without_both'] if scenario=='all' else [scenario]):
            z=g[mask(g,view)]
            for target in ['centroid','peak_transfer']:
                a=z.copy()
                if target=='peak_transfer':a['centroid_shift_hours']=a.peak_shift_hours
                performance.append(dict(model=model,training_scenario=scenario,evaluation=view,target=target,**score(a,a.predicted_hours)))
        for ex,z in g.groupby('example'):pernet.append(dict(model=model,training_scenario=scenario,example=ex,**score(z,z.predicted_hours)))
    pd.DataFrame(pernet).to_csv(OUT/'per_network_scores.csv',index=False)
    pd.DataFrame(performance).to_csv(OUT/'scores.csv',index=False)
    rng=np.random.default_rng(SEED);draws=[];fail=[];boundaries=[]
    groups={ex:g for ex,g in d.groupby('example')}
    for draw in range(boot):
        chosen=rng.choice(NETWORKS,len(NETWORKS),replace=True)
        sample=pd.concat([groups[ex].assign(example=f'draw_{i}') for i,ex in enumerate(chosen)],ignore_index=True)
        for model in ['mixed','wls']:
            try:
                b,info=fit(sample,model);draws.extend(dict(draw=draw,model=model,term=t,coefficient_hours=v) for t,v in zip(TERMS,b));boundaries.append(dict(draw=draw,model=model,boundary=info.get('boundary',False)))
            except (ValueError,np.linalg.LinAlgError) as err:fail.append(dict(draw=draw,model=model,error=str(err)))
        if (draw+1)%250==0:print('bootstrap',draw+1,flush=True)
    draws=pd.DataFrame(draws);draws.to_csv(OUT/'bootstrap_coefficients.csv',index=False);pd.DataFrame(fail,columns=['draw','model','error']).to_csv(OUT/'bootstrap_failures.csv',index=False);pd.DataFrame(boundaries).to_csv(OUT/'bootstrap_diagnostics.csv',index=False)
    summary=[];w=weights(d)
    for model in ['mixed','wls']:
        for i,c in enumerate(COLS,1):
            a=draws[(draws.model==model)&(draws.term==c)].coefficient_hours.to_numpy()*60
            loo=coef[(coef.scenario=='all')&(coef.model==model)&(coef.held!='full')&(coef.term==c)].coefficient_hours.to_numpy()*60
            b=full[model][i]*60;low,high=np.quantile(a,[.025,.975]);q1=quantile(d[c],w,.25);q3=quantile(d[c],w,.75)
            delta=np.log(2)-np.log(1.5) if c.startswith('log1p') else 1 if c=='domain_size' else .1
            summary.append(dict(model=model,term=c,coefficient_minutes=b,pointwise_low=low,pointwise_high=high,bootstrap_successes=len(a),bootstrap_same_sign=np.mean(np.sign(a)==np.sign(b)),lono_min=loo.min(),lono_max=loo.max(),lono_same_sign=np.mean(np.sign(loo)==np.sign(b)),q25=q1,q75=q3,iqr_effect_minutes=b*(q3-q1),iqr_low=low*(q3-q1),iqr_high=high*(q3-q1),reference_delta_transformed=delta,reference_effect_minutes=b*delta,reference_low=low*delta,reference_high=high*delta))
    pd.DataFrame(summary).to_csv(OUT/'effects_uncertainty.csv',index=False)
    residual=pred[pred.scenario.eq('all')].merge(d[['example','state']+COLS],on=['example','state'],validate='many_to_one');residual['error_minutes']=60*(residual.predicted_hours-residual.centroid_shift_hours);residual.to_parquet(OUT/'residual_diagnostics.parquet',index=False)
    assert all(sha(ROOT/k)==v for k,v in hashes.items()),'Source changed during run'
    manifest=json.loads((OUT/'manifest.json').read_text());manifest.update(status='complete',bootstrap_failures=len(fail));dump(OUT/'manifest.json',manifest)
    print('Saved',OUT,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bootstrap',type=int,default=2000);args=parser.parse_args();main(args.bootstrap)
