"""Eight regression families, width-first predictors and grouped relative-error tuning.

Prepare by default; --run explicitly starts resumable fitting. Existing responses
and production routing are unchanged. Full centroid is the primary training target.
"""
from pathlib import Path
import argparse, hashlib, json, warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct
import model_search as engine
import focused_models as focused
import cap_size_models as previous

ROOT=engine.ROOT
SOURCE=previous.OUT/'input.parquet'
OUT=ROOT/'RAPID/outputs/hydrograph_relative_sweep_20261019'
TARGETS=['centroid_shift_hours','cap80_hours']
EVALUATIONS=['centroid_shift_hours','peak_shift_hours','cap80_hours','cap70_hours','cap90_hours']
FLOORS=[1.,2.,5.]
COLUMNS=previous.panels()['no_resistance']
GRIDS={**focused.GRIDS,
 'spline':[dict(n_knots=k,alpha=a) for k in [3,5] for a in [.1,10.,100.]],
 'kernel_ridge':[dict(alpha=a,gamma=g) for a in [.1,1.,10.] for g in [.1,1.]],
 'svr':[dict(C=c,gamma=g,epsilon=.05) for c in [1.,10.,100.] for g in [.1,1.]],
 'gaussian_process':[dict(kernel=k,length_scale=l,noise=.1) for k in ['rbf','matern','linear_matern'] for l in [1.,3.]],
}


def relative_score(y,p,w,floor_minutes):
    if floor_minutes<=0:raise ValueError('Positive time floor required')
    return float(100*np.average(np.abs(np.asarray(p)-y)/np.maximum(np.abs(y),floor_minutes/60),weights=w))


# The only accepted legacy implementation used identical inputs/objectives and
# rejected every unconverged fit. Its completed checkpoints remain valid.
LEGACY_CONVERGED_SOURCE='899cc2b1fb652d9a55e2ab26ba69b43ef307f79961eae84cb8351f4b3984a977'
SVR_ITERATION_LIMITS=(100000,1000000,5000000)

class FitConvergenceError(RuntimeError):
    pass


def fit_converged(model,x,y,w,family):
    attempts=[]
    limits=SVR_ITERATION_LIMITS if family=='svr' else (None,)
    for limit in limits:
        if limit is not None:model.set_params(max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            if family=='gaussian_process':model.fit(x,y)
            else:model.fit(x,y,sample_weight=w)
        converged=not any(issubclass(a.category,ConvergenceWarning) for a in caught)
        if family=='svr':converged=converged and model.fit_status_==0
        attempts.append(dict(max_iter=limit,converged=converged,warnings=[str(a.message) for a in caught]))
        if converged:return attempts
        print(f'{family}: iteration limit {limit} did not converge; retrying or rejecting candidate',flush=True)
    raise FitConvergenceError(f'{family} did not converge after attempts: {attempts}')


def resume_design(out,design):
    design=json.loads(json.dumps(design))
    path=out/'design.json'
    if not path.exists():return design
    stored=json.loads(path.read_text())
    plain={k:v for k,v in stored.items() if k!='compatible_checkpoint_fingerprints'}
    if plain==design:return stored
    if not any((out/'checkpoints').glob('**/*.json')):return design
    old=plain['sources'].get(str(Path(__file__)))
    candidate=json.loads(json.dumps(plain))
    candidate['sources'][str(Path(__file__))]=design['sources'][str(Path(__file__))]
    if old!=LEGACY_CONVERGED_SOURCE or candidate!=design:
        raise ValueError('Existing checkpoints have an incompatible design; select a fresh output directory')
    legacy_fp=hashlib.sha256(json.dumps(stored,sort_keys=True).encode()).hexdigest()
    allowed=set(stored.get('compatible_checkpoint_fingerprints',[]))|{legacy_fp}
    for p in (out/'checkpoints').glob('**/*.json'):
        if json.loads(p.read_text())['fingerprint'] not in allowed:
            raise ValueError(f'Unrecognized checkpoint fingerprint: {p}')
    engine.dump(out/f'design_before_convergence_fix_{legacy_fp[:12]}.json',stored)
    design['compatible_checkpoint_fingerprints']=sorted(allowed)
    print('Preserving converged legacy checkpoints; only the convergence retry implementation changed.',flush=True)
    return design


def valid_fingerprints(design):
    return {hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest(),*design.get('compatible_checkpoint_fingerprints',[])}


def fit_predict(train,test,target,family,params,jobs=1):
    """Fit native objectives; all choices selected externally by relative error."""
    y=train[target].to_numpy(float);w=engine.weights(train)
    pre=engine.Preprocess(COLUMNS,None,None).fit(train,y)
    mean=np.average(y,weights=w);sd=np.sqrt(np.average((y-mean)**2,weights=w))
    audit=dict(kept=pre.kept,dropped=pre.dropped)
    if not pre.kept or sd<1e-14:return np.full(len(test),mean),dict(audit,constant=True)
    x,xt=pre.transform(train),pre.transform(test);z=(y-mean)/sd
    if family=='spline':
        spline=SplineTransformer(n_knots=params['n_knots'],degree=3,include_bias=False,extrapolation='linear')
        x=spline.fit_transform(x);xt=spline.transform(xt)
        scale=StandardScaler().fit(x,sample_weight=w);x=scale.transform(x);xt=scale.transform(xt)
        model=Ridge(alpha=params['alpha'])
    elif family=='kernel_ridge':model=KernelRidge(kernel='rbf',**params)
    elif family=='svr':model=SVR(kernel='rbf',max_iter=100000,**params)
    elif family=='gaussian_process':
        smooth=RBF(params['length_scale'],length_scale_bounds='fixed') if params['kernel']=='rbf' else Matern(params['length_scale'],length_scale_bounds='fixed',nu=1.5)
        kernel=DotProduct(sigma_0=1.,sigma_0_bounds='fixed')+smooth if params['kernel']=='linear_matern' else smooth
        # GP has no sample_weight fit API. Observation-noise variance inverse to
        # network-balanced weight gives a weighted Gaussian likelihood. This is
        # a modelling choice, not generic sample weighting or physical uncertainty.
        model=GaussianProcessRegressor(kernel=kernel,alpha=params['noise']/w,optimizer=None,normalize_y=False)
        audit['gp_weighting']='Noise variance = standardized noise / equal-example weight; fixed kernels; no uncertainty calibration claimed.'
    else:model=engine.estimator(family,params,jobs)
    attempts=fit_converged(model,x,z,w,family)
    pred=model.predict(xt)*sd+mean
    if not np.isfinite(pred).all():raise ValueError('Nonfinite prediction')
    return pred,dict(audit,warnings=attempts[-1]['warnings'],convergence_attempts=attempts)


def prepare(out):
    d=pd.read_parquet(SOURCE)
    assert len(d)==2170 and not d.duplicated(['example','state']).any()
    assert np.isfinite(d[COLUMNS+EVALUATIONS]).all().all()
    parent_design=json.loads((previous.OUT/'design.json').read_text())
    for p,h in parent_design['sources'].items():
        if engine.sha(Path(p))!=h:raise ValueError(f'Previous input provenance changed: {p}')
    sources=[SOURCE,previous.OUT/'design.json',Path(__file__),Path(engine.__file__),Path(previous.__file__),Path(focused.__file__)]
    design=dict(columns=COLUMNS,targets=TARGETS,evaluation_targets=EVALUATIONS,grids=GRIDS,floors_minutes=FLOORS,
                primary_target='centroid_shift_hours',primary_floor_minutes=1,
                tuning='Equal-network mean absolute relative error with denominator max(abs(observed hours), floor minutes / 60); inner grouped out-of-fold predictions. Same inner predictions reused across floors.',
                fitting='Native estimator objectives, equal-example weights; GP uses inverse-weight observation-noise variance. No claim that every estimator fits percentage loss.',
                validation=parent_design['validation'],limitations=[parent_design['limitations'],'Repeated use of outer-fold results for research means these are exploratory comparisons, not an untouched final test.'],
                gp='Fixed RBF, Matern(1.5), linear+Matern kernels; standardized noise variance .1; length scales 1 and 3; kernels selected via grouped inner error; no likelihood hyperparameter optimization.',
                resistance='Omitted from width-first core; optional addition for a later shortlisted comparison.',
                versions=dict(sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__),sources={str(p):engine.sha(p) for p in sources})
    out.mkdir(parents=True,exist_ok=True)
    design=resume_design(out,design)
    engine.dump(out/'design.json',design);d.to_parquet(out/'input.parquet',index=False)
    plan=[]
    for scenario in ['all','without54']:
        cohort=d if scenario=='all' else d[d.example.ne('54')]
        for validation in ['network','within']:
            folds=previous.outer_splits(cohort,validation)
            for target in TARGETS:
                for family,grid in GRIDS.items():
                    inner=sum(len(previous.inner_splits(tr,validation))*len(grid) for _,tr,_ in folds)
                    plan.append(dict(scenario=scenario,validation=validation,target=target,model=family,outer_folds=len(folds),test_rows=sum(len(te) for _,_,te in folds),inner_fits=inner,outer_fits_min=len(folds),outer_fits_max=len(folds)*len(FLOORS)))
    plan=pd.DataFrame(plan);plan.to_csv(out/'run_plan.csv',index=False)
    print(f'Prepared {len(d)} comparisons, {len(GRIDS)} model families, 48 settings per target; no fits unless --run.',flush=True)
    print('Full suite fits:',int((plan.inner_fits+plan.outer_fits_min).sum()),'to',int((plan.inner_fits+plan.outer_fits_max).sum()),flush=True)
    return d,design


def summarize(out,design):
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest();rows=[];settings=[]
    for path in (out/'checkpoints').glob('*/*/*/*.json'):
        r=json.loads(path.read_text());assert r['fingerprint'] in valid_fingerprints(design)
        common={k:r[k] for k in ['scenario','validation','held','target','model']}
        for result in r['results']:
            settings.append(dict(**common,floor_minutes=result['floor_minutes'],params=json.dumps(result['params']),inner_score=result['inner_score']))
            rows.extend(dict(**common,floor_minutes=result['floor_minutes'],**row) for row in result['predictions'])
    if not rows:return
    pred=pd.DataFrame(rows);pred.to_parquet(out/'predictions.parquet',index=False)
    pd.DataFrame(settings).to_csv(out/'chosen_hyperparameters.csv',index=False)
    data=pd.read_parquet(out/'input.parquet');plan=pd.read_csv(out/'run_plan.csv');scores=[]
    keys=['scenario','validation','target','model','floor_minutes']
    for values,g in pred.groupby(keys):
        common=dict(zip(keys,values));assert not g.duplicated(['example','state']).any()
        exp=plan[plan.scenario.eq(values[0])&plan.validation.eq(values[1])&plan.target.eq(values[2])&plan.model.eq(values[3])].iloc[0]
        common['complete']=bool(len(g)==exp.test_rows and g.held.nunique()==exp.outer_folds)
        g=g.merge(data[['example','state']+EVALUATIONS],on=['example','state'],validate='one_to_one')
        for evaluated in EVALUATIONS:
            y=g[evaluated]*60
            for threshold in FLOORS:
                masks=dict(all=np.ones(len(g),bool),unchanged=y.abs().le(threshold),advance=y.lt(-threshold),delay=y.gt(threshold),outside_buffer=y.abs().gt(threshold),large_shift=y.abs().ge(60))
                for label,mask in masks.items():
                    z=g[mask]
                    if z.empty:continue
                    yy=z[evaluated].to_numpy();pp=z.predicted_hours.to_numpy()
                    for weighting,w in [('equal_example',engine.weights(z)),('pair',np.ones(len(z)))]:
                        row=dict(**common,evaluated_target=evaluated,evaluation_floor_minutes=threshold,subset=label,weighting=weighting,n=len(z),examples=z.example.nunique(),bias_minutes=float(np.average((pp-yy)*60,weights=w)),relative_error_pct=relative_score(yy,pp,w,threshold),**engine.score(yy*60,pp*60,w))
                        if label=='outside_buffer':row['mape_pct']=float(100*np.average(abs((pp-yy)/yy),weights=w))
                        scores.append(row)
    pd.DataFrame(scores).to_csv(out/'scores.csv',index=False)


def run(d,design,out,args):
    fp=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    for scenario in args.scenarios:
        cohort=d if scenario=='all' else d[d.example.ne('54')]
        for validation in args.validations:
            for held,tr,te in previous.outer_splits(cohort,validation):
                for target in args.targets:
                    for family in args.models:
                        path=out/'checkpoints'/scenario/validation/held/f'{target}__{family}.json'
                        if path.exists():
                            assert json.loads(path.read_text())['fingerprint'] in valid_fingerprints(design);continue
                        trials=[];splits=previous.inner_splits(tr,validation)
                        for params in GRIDS[family]:
                            predicted=np.full(len(tr),np.nan)
                            try:
                                for ti,vi in splits:
                                    predicted[vi],_=fit_predict(tr.iloc[ti],tr.iloc[vi],target,family,params,args.jobs)
                            except FitConvergenceError as exc:
                                trials.append(dict(params=params,valid=False,reason=str(exc)))
                                continue
                            assert np.isfinite(predicted).all()
                            trials.append(dict(params=params,valid=True,scores={str(f):relative_score(tr[target].to_numpy(),predicted,engine.weights(tr),f) for f in FLOORS}))
                        valid=[t for t in trials if t['valid']]
                        if not valid:raise FitConvergenceError(f'No converged candidates for {scenario}/{validation}/{held}/{target}/{family}; earlier checkpoints preserved')
                        fitted={};failed={};results=[]
                        for floor in FLOORS:
                            for best in sorted(valid,key=lambda a:a['scores'][str(floor)]):
                                key=json.dumps(best['params'],sort_keys=True)
                                if key in failed:continue
                                if key not in fitted:
                                    try:fitted[key]=fit_predict(tr,te,target,family,best['params'],args.jobs)
                                    except FitConvergenceError as exc:
                                        failed[key]=str(exc);continue
                                prediction,audit=fitted[key]
                                rows=[dict(example=r.example,state=r.state,observed_hours=float(getattr(r,target)),predicted_hours=float(p)) for r,p in zip(te.itertuples(),prediction)]
                                results.append(dict(floor_minutes=floor,params=best['params'],inner_score=best['scores'][str(floor)],audit=audit,predictions=rows))
                                break
                            else:raise FitConvergenceError(f'No converged outer fit for {held}/{target}/{family}; earlier checkpoints preserved')
                        engine.dump(path,dict(fingerprint=fp,scenario=scenario,validation=validation,held=held,target=target,model=family,trials=trials,outer_convergence_failures=failed,results=results))
                        print(scenario,validation,held,target,family,'saved',flush=True)
            summarize(out,design)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--summarize-only',action='store_true');p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--models',nargs='+',choices=list(GRIDS),default=list(GRIDS))
    p.add_argument('--targets',nargs='+',choices=TARGETS,default=TARGETS)
    p.add_argument('--scenarios',nargs='+',choices=['all','without54'],default=['all','without54'])
    p.add_argument('--validations',nargs='+',choices=['network','within'],default=['network','within'])
    args=p.parse_args()
    if args.jobs<1:p.error('--jobs must be positive')
    d,design=prepare(args.output)
    if args.run and not args.summarize_only:run(d,design,args.output,args)
    summarize(args.output,design)

if __name__=='__main__':main()
