"""Resistance versus geometry on identical records, with example-held-out folds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analyze import TARGETS,weights,metrics,fit_predict
from scopes import OUT,ROOT
from build import digest


def geometry(scope):
    return [scope+'_exposed_length_change',scope+'_mean_length_fraction',scope+'_exposed_sd_change',scope+'_exposure',
            scope+'_ratio_width_mean',scope+'_parent_dominance_mean',scope+'_delta_expected_splits']
R=['B_K_log_a1','B_weighted_K_log_a1']
PANELS={
 'R_only':['B_K_log_a1'],
 'R_weighted_only':['B_weighted_K_log_a1'],
 'R_context':R,
 'geometry':geometry('B'),
 'R_plus_L':R+geometry('B')[:4],
 'R_plus_LW':R+geometry('B')[:6],
 'R_plus_LWT':R+geometry('B'),
 'R065_context':['B_K_log_a065','B_weighted_K_log_a065'],
 'R065_plus_LWT':['B_K_log_a065','B_weighted_K_log_a065']+geometry('B'),
}
PAIRED_PANELS={'R_C':R,'U_geometry':geometry('U'),'C_geometry':geometry('C'),
               'R_C_plus_U':R+geometry('U'),'R_C_plus_C':R+geometry('C')}


def main():
    d=pd.read_parquet(OUT/'linked_responses.parquet')
    graphs=pd.read_csv(OUT/'domain_graphs.csv',dtype={'example':str})
    up=graphs[graphs.scope.eq('U')][['example','state','parent_graph','child_graph']]
    cp=graphs[graphs.scope.eq('C')][['example','state','parent_graph','child_graph']]
    uc=up.merge(cp,on=['example','state'],validate='one_to_one',suffixes=('_U','_C'))
    uc['distinct_U_C']=(uc.parent_graph_U!=uc.parent_graph_C)|(uc.child_graph_U!=uc.child_graph_C)
    d=d.merge(uc[['example','state','distinct_U_C']],on=['example','state'],how='left',validate='one_to_one')
    eligible=d.B_available & d.B_K_log_a1.notna() & d.B_K_log_a065.notna()
    excluded=d[~eligible][['example','state','B_available','B_K_valid_a1','B_K_valid_a065']]
    excluded.to_csv(OUT/'benchmark_exclusions.csv',index=False)
    cohorts={
        'partial':d[eligible&d.comparison.eq('partial collapse')],
        'including_centerlines':d[eligible],
        'paired_U_C':d[eligible&d.distinct_U_C.fillna(False)&d.comparison.eq('partial collapse')],
    }
    predrows=[];folds=[];cohortrows=[];coeff=[]
    for name,source in cohorts.items():
        panels=PAIRED_PANELS if name=='paired_U_C' else PANELS
        # Same records for all panels in this comparison; only conditional no-split means may be imputed.
        required=set(c for cols in panels.values() for c in cols if 'dominance' not in c)
        assert source[list(required)].notna().all().all()
        for include54 in [True,False]:
            data=source if include54 else source[source.example.ne('54')]
            run=name+'/'+('all' if include54 else 'without54')
            for ex,g in data.groupby('example'):cohortrows.append(dict(run=run,example=ex,n=len(g)))
            for held in sorted(data.example.unique()):
                train=data[data.example.ne(held)];test=data[data.example.eq(held)]
                assert not set(train.example)&set(test.example)
                mean=np.average(train[TARGETS],axis=0,weights=weights(train))
                predictions={'zero':np.zeros((len(test),len(TARGETS))),'training_mean':np.tile(mean,(len(test),1))}
                for panel,cols in panels.items():
                    estimate,kept,coefs=fit_predict(train,test,cols);predictions[panel]=estimate
                    folds.append(dict(run=run,panel=panel,held_example=held,train_n=len(train),test_n=len(test),features=' | '.join(kept)))
                    for j,target in enumerate(TARGETS):
                        for k,feature in enumerate(kept):coeff.append(dict(run=run,panel=panel,held_example=held,response=target,feature=feature,coefficient=coefs[j,k]))
                for panel,estimate in predictions.items():
                    for j,target in enumerate(TARGETS):
                        block=test[['example','state']].copy();block['run']=run;block['panel']=panel;block['response']=target
                        block['observed']=test[target].to_numpy();block['predicted']=estimate[:,j];predrows.append(block)
            print(run,'held-example fits complete',flush=True)
    pred=pd.concat(predrows,ignore_index=True);assert not pred.duplicated(['run','panel','response','example','state']).any()
    pred.to_parquet(OUT/'predictions.parquet',index=False);pd.DataFrame(folds).to_csv(OUT/'fold_audit.csv',index=False)
    pd.DataFrame(coeff).to_csv(OUT/'fold_coefficients.csv',index=False);pd.DataFrame(cohortrows).to_csv(OUT/'cohort_counts.csv',index=False)
    score=[]
    for (run,panel,target),g in pred.groupby(['run','panel','response']):
        for weighting,w in [('equal_example',weights(g)),('pair',np.ones(len(g)))]:
            score.append(dict(run=run,panel=panel,response=target,weighting=weighting,n=len(g),examples=g.example.nunique(),**metrics(g.observed,g.predicted,w)))
        for ex,gg in g.groupby('example'):
            score.append(dict(run=run,panel=panel,response=target,weighting='example_'+ex,n=len(gg),examples=1,**metrics(gg.observed,gg.predicted,np.ones(len(gg)))))
    pd.DataFrame(score).to_csv(OUT/'scores.csv',index=False)
    # Individual relationships across all six targets; not used for selection.
    associations=[];redundancy=[]
    for cohort,source in [('partial',cohorts['partial']),('paired_U_C',cohorts['paired_U_C'])]:
        panels=PANELS if cohort=='partial' else PAIRED_PANELS
        cols=list(dict.fromkeys(c for v in panels.values() for c in v))
        cor=source[cols].corr(method='spearman')
        for i,c in enumerate(cols):
            for other in cols[i+1:]:
                if abs(cor.loc[c,other])>.85:redundancy.append(dict(cohort=cohort,feature=c,other=other,rho=cor.loc[c,other]))
        for feature in cols:
            for target in TARGETS:
                for ex,g in [('pooled',source),*list(source.groupby('example'))]:
                    z=g[[feature,target]].dropna();rho=spearmanr(z[feature],z[target]).statistic if len(z)>2 and z.nunique().min()>1 else np.nan
                    associations.append(dict(cohort=cohort,example=ex,feature=feature,response=target,n=len(z),rho=rho))
    pd.DataFrame(associations).to_csv(OUT/'associations.csv',index=False);pd.DataFrame(redundancy).to_csv(OUT/'redundancy.csv',index=False)
    # Preserve the primary window; sensitivity evaluates the same out-of-example predictions.
    sensitivity=pd.read_parquet(ROOT/'RAPID/outputs/hydrograph_response/sensitivity_responses.parquet');ss=[]
    for win in ['1%','2%']:
        for target,g in pred.groupby('response'):
            alt=sensitivity[sensitivity.window.eq(win)][['example','state',target]]
            v=g.merge(alt,on=['example','state'],validate='many_to_one');assert len(v)==len(g)
            for (run,panel),gg in v.groupby(['run','panel']):
                ss.append(dict(window=win,run=run,panel=panel,response=target,**metrics(gg[target],gg.predicted,weights(gg))))
    pd.DataFrame(ss).to_csv(OUT/'sensitivity_scores.csv',index=False)
    metadata=dict(panels=PANELS,paired_panels=PAIRED_PANELS,targets=TARGETS,primary=['peak_shift_hours','peak_excess_change_pct'],
        validation='Leave entire examples out; refit with 54 absent from training and test. Fixed Ridge alpha=10; no tuning. Training-only imputation/scaling, equal example weights.',
        cohorts={k:dict(n=len(v),examples=v.example.nunique()) for k,v in cohorts.items()},excluded=len(excluded),
        resistance_context='B_log=-log10(Kc/Kp); B_weighted_log=(parent unique domain length/parent unique network length)*parent entry exposure*B_log',
        U_C_convention='Same edits and identical cohort. Only pairs with distinct U/C graph domains; U may be open, with conditional in-domain geometry allocation; never use open-U resistance.',
        limitations=['Exploratory fixed-panel comparison, not a final model-selection result','Resistance numerical energy gap is bounded to 1e-8, not exact arithmetic','Fixed-law resistance assumptions are not independently measured hydraulics','Crest-switch diagnostics and response metrics never enter predictor columns'],
        hashes={str(OUT/'linked_responses.parquet'):digest(OUT/'linked_responses.parquet'),str(Path(__file__)):digest(__file__)})
    (OUT/'comparison_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
