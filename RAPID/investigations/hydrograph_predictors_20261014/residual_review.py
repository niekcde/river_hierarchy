"""Descriptive held-example error review; no model fitting or variable selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import model_search as engine
import focused_models as focused

ROOT=engine.ROOT
SOURCE=ROOT/'RAPID/outputs/hydrograph_context_timing_20261018'
OUT=ROOT/'RAPID/outputs/hydrograph_residual_review_20261018'
MODEL='length_context__sd__log1p__elastic_net'
BASE='baseline__sd__log1p__elastic_net'
HASHES={str(Path(__file__)):engine.sha(__file__)}
def read(p):
 p=Path(p);HASHES[str(p)]=engine.sha(p)
 return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,dtype={'example':str}) if p.suffix=='.csv' else json.loads(p.read_text())

def classify(data):
 plan=read(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json');rows=[]
 for c in plan['cases']:
  ex=c['example'];jobs={('fully_collapsed' if j['state_id']=='single_centerline' else j['state_id']):j for j in c['jobs']}
  base=Path(jobs['S000_base']['parameters']).parents[3]
  units=read(base/'hierarchy/unit_summary.csv').set_index('unit_id')
  for r in data[data.example.eq(ex)].itertuples():
   j=jobs[r.state];families={a.split('/')[0] for a in j['aliases']};primary=j['aliases'][0].split('/')[0]
   ids=[] if pd.isna(r.selected_unit_ids) else [int(float(s)) for s in str(r.selected_unit_ids).split(',') if s]
   u=units.loc[ids];nested=bool(len(u) and ((u.n_children>0)|(u.depth_from_root>0)).any())
   member=bool(r.is_compound_member)
   if 'compound_bubbles' in families:kind='Full compound'
   elif primary=='island':kind='Island'
   elif member:kind='Compound member'
   elif nested:kind='Nested'
   elif len(u) and u['class'].eq('simple_bifurcation_confluence_pair').all():kind='Simple'
   else:kind='Unclassified'
   rows.append(dict(example=ex,state=r.state,edit_type=kind,primary_family=primary,alias_families=' | '.join(sorted(families)),
     has_island_alias='island' in families,has_nested_structure=nested,containing_compound=member,
     selected_unit_count=len(ids),max_depth=float(u.depth_from_root.max()) if len(u) else np.nan,
     compound_ids=','.join(map(str,sorted(u.compound_bubble_id.dropna().astype(int).unique())))))
 return pd.DataFrame(rows)

def summarize(g,balanced=False):
 w=engine.weights(g) if balanced else np.ones(len(g))
 a=lambda x:float(np.average(x,weights=w))
 e=g.error_min.to_numpy();y=g.observed_hours.to_numpy()*60
 zero=a(abs(y));mae=a(abs(e))
 return dict(n=len(g),examples=g.example.nunique(),bias_min=a(e),mae_min=mae,
  rmse_min=np.sqrt(a(e**2)),median_abs_error_min=float(np.median(abs(e))),
  p90_abs_error_min=float(np.quantile(abs(e),.9)),mean_abs_shift_min=zero,
  mae_vs_zero=mae/zero if zero>0 else np.nan,baseline_mae_min=a(abs(g.baseline_error_min)),
  positive_shifts=int((y>0).sum()),observed_min_min=y.min(),observed_max_min=y.max())

def rho(x,y):
 z=pd.DataFrame({'x':x,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 return float(spearmanr(z.x,z.y).statistic) if len(z)>=10 and z.x.nunique()>1 and z.y.nunique()>1 else np.nan

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 pred=read(SOURCE/'predictions.parquet');geom=read(engine.SOURCE/'linked_responses.parquet')
 a=pred[pred.pipeline.eq(MODEL)].copy();b=pred[pred.pipeline.eq(BASE)][['scenario','example','state','predicted_hours']].rename(columns={'predicted_hours':'baseline_prediction'})
 a=a.merge(b,on=['scenario','example','state'],validate='one_to_one')
 d=a.merge(geom,on=['example','state'],validate='many_to_one',how='left',indicator=True)
 d[['scenario','example','state','_merge']].to_csv(OUT/'join_audit.csv',index=False)
 assert d._merge.eq('both').all();d=d.drop(columns='_merge')
 np.testing.assert_allclose(d.observed_hours,d.peak_shift_hours)
 labels=classify(d[d.scenario.eq('all')]);assert not labels.duplicated(['example','state']).any()
 labels.to_csv(OUT/'classification_audit.csv',index=False)
 d=d.merge(labels,on=['example','state'],validate='many_to_one')
 d['error_min']=60*(d.predicted_hours-d.observed_hours);d['abs_error_min']=abs(d.error_min)
 d['baseline_error_min']=60*(d.baseline_prediction-d.observed_hours)
 # Check inputs outside the training-example range, not a joint density estimate.
 design=read(SOURCE/'design.json');spec=next(s for s in design['specifications'] if s['name']==MODEL)
 d=focused.transform(d);support=[]
 for scenario,g in d.groupby('scenario'):
  for ex,test in g.groupby('example'):
   train=g[g.example.ne(ex)];flags=pd.DataFrame(index=test.index)
   for c in spec['columns']:
    lo,hi=train[c].min(),train[c].max();flags[c]=(test[c]<lo-1e-12)|(test[c]>hi+1e-12)
    support.append(dict(scenario=scenario,example=ex,feature=c,outside_fraction=float(flags[c].mean()),train_min=lo,train_max=hi))
   d.loc[test.index,'outside_input_range']=flags.any(axis=1)
 pd.DataFrame(support).to_csv(OUT/'training_range_audit.csv',index=False)
 tables=[];types=[];cross=[]
 for scenario,g in d.groupby('scenario'):
  means=g.groupby('example').error_min.apply(lambda x:np.mean(x*x));den=means.sum()
  for ex,z in g.groupby('example'):
   tables.append(dict(scenario=scenario,example=ex,**summarize(z),
     equal_example_squared_error_share_pct=100*means[ex]/den,
     outside_input_range_fraction=z.outside_input_range.mean()))
  for kind,z in g.groupby('edit_type'):types.append(dict(scenario=scenario,edit_type=kind,**summarize(z,True)))
  for (ex,kind),z in g.groupby(['example','edit_type']):cross.append(dict(scenario=scenario,example=ex,edit_type=kind,**summarize(z)))
 pd.DataFrame(tables).to_csv(OUT/'network_errors.csv',index=False)
 pd.DataFrame(types).to_csv(OUT/'edit_type_errors.csv',index=False)
 pd.DataFrame(cross).to_csv(OUT/'network_edit_errors.csv',index=False)
 keep=['scenario','example','state','edit_type','aliases','observed_hours','predicted_hours','error_min','abs_error_min',
       'outside_input_range','has_nested_structure','has_island_alias','compound_ids']
 d[keep].sort_values('abs_error_min',ascending=False).groupby('scenario').head(30).to_csv(OUT/'worst_cases.csv',index=False)
 # Existing omitted geometry only; never response-derived quantities as candidates.
 measures=['route_length_mean','route_length_sd','width_mean','route_length_skew','route_length_min','route_length_max','branch_length_contrast_mean',
 'branch_length_contrast_sd','dominance_sd','split_entropy_sd','effective_branches_sd','expected_splits',
 'cycle_rank','bifurcations','bifurcation_density','split_impact_sum','split_impact_sd','split_exposure_sd',
 'width_harmonic','area_elongation','total_length']
 candidates=[f'B_{rep}_{m}' for m in measures for rep in ['parent','delta']]
 details=[];summary=[]
 for scenario,g in d.groupby('scenario'):
  for c in candidates:
   for target in ['error_min','abs_error_min']:
    rs=[]
    for ex,z in g.groupby('example'):
     r=rho(z[c],z[target]);details.append(dict(scenario=scenario,feature=c,error_type=target,example=ex,n=int(z[[c,target]].dropna().shape[0]),rho=r))
     if np.isfinite(r):rs.append(r)
    # Pooled rank association and within-example associations are intentionally separate.
    summary.append(dict(scenario=scenario,feature=c,error_type=target,coverage=float(g[c].notna().mean()),
       pooled_rho=rho(g[c],g[target]),examples_with_variation=len(rs),
       median_within_rho=float(np.median(rs)) if rs else np.nan,
       positive_examples=sum(r>0 for r in rs),negative_examples=sum(r<0 for r in rs),
       examples_abs_rho_ge03=sum(abs(r)>=.3 for r in rs)))
 pd.DataFrame(details).to_csv(OUT/'residual_associations_by_example.csv',index=False)
 pd.DataFrame(summary).to_csv(OUT/'residual_associations.csv',index=False)
 strata=[]
 for (scenario,kind,nested),g in d.groupby(['scenario','edit_type','has_nested_structure']):
  strata.append(dict(scenario=scenario,edit_type=kind,nested=nested,**summarize(g,True)))
 pd.DataFrame(strata).to_csv(OUT/'nested_strata_errors.csv',index=False)
 # Within-edit-type checks avoid attributing between-type differences to a feature.
 controlled=[]
 for (scenario,kind),g in d.groupby(['scenario','edit_type']):
  for c in candidates:
   rs=[rho(z[c],z.error_min) for _,z in g.groupby('example')]
   rs=[r for r in rs if np.isfinite(r)]
   controlled.append(dict(scenario=scenario,edit_type=kind,feature=c,examples_with_variation=len(rs),
       median_within_rho=float(np.median(rs)) if rs else np.nan,positive_examples=sum(r>0 for r in rs),negative_examples=sum(r<0 for r in rs)))
 pd.DataFrame(controlled).to_csv(OUT/'within_type_associations.csv',index=False)
 # Repeated member edits share outer-domain predictors: aggregate by parent graph
 # as a descriptive robustness check, never claim independent row-level significance.
 graphs=read(engine.SOURCE/'domain_graphs.csv')[['example','state','scope','parent_graph']]
 dd=d.merge(graphs,left_on=['example','state','B_scope'],right_on=['example','state','scope'],validate='many_to_one')
 assert len(dd)==len(d)
 grouped=dd.groupby(['scenario','example','parent_graph'])[candidates+['error_min','abs_error_min']].median().reset_index()
 domain_rows=[]
 for scenario,g in grouped.groupby('scenario'):
  for c in candidates:
   rs=[rho(z[c],z.error_min) for _,z in g.groupby('example')];rs=[r for r in rs if np.isfinite(r)]
   domain_rows.append(dict(scenario=scenario,feature=c,outer_domains=len(g),examples_with_variation=len(rs),
       median_within_rho=float(np.median(rs)) if rs else np.nan,positive_examples=sum(r>0 for r in rs),negative_examples=sum(r<0 for r in rs)))
 pd.DataFrame(domain_rows).to_csv(OUT/'domain_aggregated_associations.csv',index=False)
 # Whole-network parent geometry is constant within each example: only 15 points.
 profile=[]
 for ex,g in d[d.scenario.eq('all')].groupby('example'):
  row=dict(example=ex,n=len(g),**{c:float(g[c].iloc[0]) for c in geom if c.startswith('N_parent_')})
  row.update(summarize(g));profile.append(row)
 pd.DataFrame(profile).to_csv(OUT/'network_profiles.csv',index=False)
 d.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
 engine.dump(OUT/'manifest.json',dict(model=MODEL,baseline=BASE,records=len(d),primary_records=int(d.scenario.eq('all').sum()),
  classification='One canonical row per response. Any compound_bubbles-family alias -> full compound; otherwise primary family island -> island; otherwise compound membership -> compound member; otherwise n_children>0 or depth>0 -> nested; otherwise simple class -> simple. Nested/island alias flags retained separately. These categories overlap structurally; precedence avoids double counting.',
  residual_sign='predicted minus observed; positive means prediction is too late (or insufficiently early).',
  diagnostics='Descriptive only, no new model fits. Spearman pooled and per-example; no row-level p-values, no causal/conditional importance claims. abs-error association describes difficulty, not necessarily missing signed effect. Network profiles have only 15 independent examples.',
  caveats=['Model chosen after viewing this cohort; further hypothesis generation is exploratory.',
   'Rows share parents and compound domains; row counts are not independent replicates.',
   'Edit-type summaries equal-weight examples represented in each category; category network mixtures differ.',
   'Training-range check is univariate, not joint distribution coverage.'],source_hashes=HASHES))
 print(pd.DataFrame(tables).query("scenario=='all'").sort_values('mae_min',ascending=False).to_string(index=False))
 print(pd.DataFrame(types).query("scenario=='all'").to_string(index=False))
if __name__=='__main__':main()
