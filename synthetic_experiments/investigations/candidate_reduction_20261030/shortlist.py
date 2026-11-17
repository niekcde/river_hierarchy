"""Transparent analyst-curated pre-RF shortlist with robustness diagnostics."""
from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import audit
O=audit.OUT;S=audit.SOURCE
# Not optimized by an RF or a target-score cutoff. Alternative entries are swaps,
# context entries protect concepts the fixed-topology grid cannot assess.
SPEC=[
('delta_G_manning','primary','Manning ΔG','Joint geometry','Physically specified length–width–allocation–slope combination; strongest Manning centroid association; retain beside its length component to test added information.'),
('B_delta_route_length_mean','primary','Mean route-length change','Distance change','Direct width-allocation-weighted change in distance; clear physical baseline; canonical metres rather than duplicate kilometres.'),
('B_delta_route_length_sd','primary','Route-spread change','Route heterogeneity','Well-defined even when parent spread is zero; preserves spread change without an undefined ratio.'),
('B_delta_area_proxy','primary','Channel-area change','Length × width','Strongest peak association among complete candidates; not a pure width descriptor.'),
('B_parent_route_length_mean','primary','Parent mean route length','Absolute scale','Provides domain scale and allows a model to distinguish relative and absolute changes; strong BASED association.'),
('domain_size','primary','Affected-domain size','Affected scale','Width-allocation-weighted length of the selected source unit in kilometres; preserves affected extent separately from surrounding parent-domain size and is strong under BASED.'),
('B_parent_width_mean','primary','Parent mean width','Absolute width','Separates width scale from length; weak marginal association is not grounds to remove an interacting scale variable.'),
('path_parent_slope_mean','primary','Parent mean slope','Slope scale','Interpretable length-weighted slope reference; alternative to inverse-square-root slope transform.'),
('path_delta_slope_mean','primary','Mean slope change','Slope change','Strong peak rank association with weaker linear association; retain for nonlinear response tests.'),
('B_parent_width_cv','primary','Parent width CV','Width heterogeneity','Dimensionless width heterogeneity; defined for single channels and uniform widths, unlike some CV ratios.'),
('B_parent_dominance_mean','primary','Parent width dominance','Allocation','Direct width-based splitting asymmetry. Weak marginal association does not exclude a length interaction; valid on bifurcating parents in this population.'),
('B_delta_route_entropy','primary','Route-entropy change','Allocation/topology','Complete difference with a defined single-route endpoint; avoids ratios masking the size of the original entropy.'),
('path_parent_signed_length_width_association','primary','Parent length–width association','Length–width pairing','Distinguishes which routes carry width; not interchangeable with dominance. Historical path harmonic-width weighting, not discharge fractions.'),
('B_parent_route_length_sd','primary','Parent route spread','Route heterogeneity','Context for a given spread change. Perfect within-type correlation with spread change in this grid does not imply one global conversion across collapse types.'),
('path_delta_equivalent_length','alternative','Equivalent-length change','Distance alternative','Strong association but nearly affine to mean-route change in this grid; test as a replacement before including both.'),
('path_parent_slope_cv','alternative','Parent slope CV','Relative heterogeneity alternative','Captures relative slope/path contrasts; coupled to length variability by preserved head loss, so not an independently established slope mechanism.'),
('log1p__B_ratio_route_length_mean','alternative','Mean route-length ratio (log1p)','Relative distance alternative','Retains the existing RF relative-distance representation for a swap test; raw ratio adds no independent ordering information.'),
('B_parent_log1p_paths','context reserve','Parent path count (log1p)','Topology context','Constant within each topology type, so within-type correlations cannot evaluate it. Reserve for more complex real geometries; not established by synthetic sensitivity.'),
('B_delta_log1p_paths','context reserve','Change in log1p path count','Topology change','Separates amount of topological simplification from parent complexity; needs real-case testing, not a claim of synthetic explanatory power.'),
]
SHORT=[r[0] for r in SPEC];LABEL={r[0]:r[2] for r in SPEC}

def savefig(fig,name):
 for ext in ['png','pdf']:fig.savefig(O/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)

def main():
 meta=json.loads((O/'provenance.json').read_text())
 for p,h in meta['sources'].items():assert audit.sha(p)==h,p
 inv=pd.read_csv(O/'inventory.csv');cov=pd.read_csv(O/'coverage_by_type.csv');pairs=pd.read_csv(O/'predictor_pairs.csv');aff=pd.read_csv(O/'affine_equalities.csv');clusters=pd.read_csv(O/'clusters.csv')
 assoc=pd.read_csv(O/'response_associations.csv');summary=pd.read_csv(O/'response_summary.csv');effects=pd.read_csv(S/'factorial_signatures.csv')
 f=pd.read_parquet(S/'features.parquet')
 raw_geometry=f[f.method.eq('variable_manning')].copy()
 for c in SHORT:f[c]=audit.clean(f[c])
 g=f[f.method.eq('variable_manning')].copy()
 short=pd.DataFrame(SPEC,columns=['variable','tier','label','mechanism','reason']).merge(inv[['variable','minimum_type_coverage','varying_types','family','definition','scope']],on='variable',validate='one_to_one')
 assert short.minimum_type_coverage.eq(100).all()
 short.to_csv(O/'shortlist.csv',index=False)
 sub=summary[summary.variable.isin(SHORT)].merge(short[['variable','tier','label']],on='variable');sub.to_csv(O/'shortlist_response_summary.csv',index=False)
 # Every candidate receives a traceable disposition. Correlation groups are evidence,
 # not universal algebraic identities or automatic reasons to discard geometry.
 parent={c:c for c in inv.variable}
 def root(c):
  while parent[c]!=c:c=parent[c]
  return c
 for r in aff.itertuples():parent[root(r.b)]=root(r.a)
 groups={}
 for c in inv.variable:groups.setdefault(root(c),[]).append(c)
 canonical={}
 for group in groups.values():
  pick=next((c for c in SHORT if c in group),sorted(group)[0])
  for c in group:canonical[c]=pick
 cl=clusters[clusters.threshold.eq(.9)].set_index('variable').cluster.to_dict();tier=dict(zip(short.variable,short.tier));reasons=dict(zip(short.variable,short.reason));rows=[]
 for r in inv.itertuples():
  reference='';reason=''
  if r.variable in SHORT:decision=tier[r.variable];reason=reasons[r.variable]
  elif r.status=='partly_defined':decision='conditional_only';reason='Not a universal RF input under current definitions; retain in conditional inventory, do not impute zero.'
  elif r.status=='globally_constant':decision='design_invariant';reason='No variation anywhere on this grid; not evidence of universal irrelevance.'
  elif canonical[r.variable]!=r.variable:decision='empirical_affine_alias';reference=canonical[r.variable];reason='Numerically affine on the full synthetic population; formula review required before transferring the equivalence to real networks.'
  elif r.status=='between_type_only':decision='topology_reserve_not_selected';reason='Fixed within each type; represented by the two selected topology-context candidates for now.'
  else:
   same=[c for c in SHORT if cl.get(c,-1)==cl.get(r.variable,-2)]
   if same:decision='correlated_alternative';reference=' | '.join(same);reason='Same conservative .90 complete-linkage group; retain formula distinction in the catalogue, not automatically in the small pool.'
   else:decision='deferred_from_small_pool';reason='Different descriptor retained in the expanded library; not needed for the current interpretable mechanism pool. No claim of zero conditional predictive value.'
  rows.append(dict(variable=r.variable,decision=decision,representative=reference,reason=reason,cluster_090=cl.get(r.variable),status=r.status,family=r.family))
 ledger=pd.DataFrame(rows);ledger.to_csv(O/'candidate_decisions.csv',index=False)
 # Sensitivity to deleting each control level, without treating design points as iid.
 # Summaries stay within type; all six Manning-resolved types included with counts.
 levels=[]
 resolved=assoc[(assoc.method=='variable_manning')&(assoc.response=='centroid_shift_minutes')&assoc.resolved].child_state.unique()
 for factor in audit.FACTORS:
  for level in sorted(g[factor].unique()):
   zz=g[g[factor]!=level]
   for state,z in zz[zz.child_state.isin(resolved)].groupby('child_state'):
    for c in SHORT:
     x=audit.clean(z[c])
     for response in audit.RESPONSES:levels.append(dict(omitted_factor=factor,omitted_level=level,child_state=state,variable=c,response=response,spearman=audit.corr(x,z[response].to_numpy(),True)))
 lev=pd.DataFrame(levels);lev.to_csv(O/'leave_level_out.csv',index=False)
 lm=lev.groupby(['omitted_factor','omitted_level','variable','response']).agg(mean_abs_spearman=('spearman',lambda x:x.abs().mean()),n_types=('spearman','count')).reset_index()
 lm.to_csv(O/'leave_level_out_summary.csv',index=False)
 # Leave one collapse type out: robustness envelope, not a sampling confidence interval.
 lot=[]
 for (method,response),z in assoc[assoc.variable.isin(SHORT)&assoc.resolved].groupby(['method','response']):
  for state in z.child_state.unique():
   for c,v in z[z.child_state!=state].groupby('variable'):lot.append(dict(method=method,response=response,omitted_type=state,variable=c,mean_abs_spearman=v.spearman.abs().mean(),n_types=v.spearman.count()))
 pd.DataFrame(lot).to_csv(O/'leave_type_out.csv',index=False)
 # Cross-method comparisons on identical six-type support, avoiding seven vs six confounding.
 common=assoc[assoc.variable.isin(SHORT)&assoc.child_state.isin(resolved)].groupby(['method','response','variable']).agg(mean_abs_spearman=('spearman',lambda x:x.abs().mean()),mean_abs_pearson=('pearson',lambda x:x.abs().mean()),n_types=('spearman','count')).reset_index()
 common.to_csv(O/'common_support_methods.csv',index=False)
 # Quantization and predictor roundoff sensitivity, explicitly diagnostic rather than
 # claiming any particular bin size is the error bound of centroid estimation.
 rounding=[];counts=[]
 allcols=inv.variable.tolist()
 for eps in [1e-8,1e-10,1e-12]:
  for state,z in raw_geometry.groupby('child_state'):
   for c in allcols:
    xx=z[c].to_numpy(float);finite=np.isfinite(xx);scale=max(float(np.nanmax(abs(raw_geometry[c]))) if np.isfinite(raw_geometry[c]).any() else 0.,1.);t=eps*scale
    x=np.round(xx/t)*t;counts.append(dict(tolerance_factor=eps,child_state=state,variable=c,varying=bool(finite.sum()>1 and np.ptp(x[finite])>t)))
 for method,z0 in f.groupby('method'):
  for state,z in z0.groupby('child_state'):
   for response in audit.RESPONSES:
    y=z[response].to_numpy();resolved_y=np.ptp(y)>(4/60 if response.startswith('centroid') else .5)
    if not resolved_y:continue
    for seconds in ([0,1,4] if response.startswith('centroid') else [0,60]):
     yy=y if seconds==0 else np.round(y/(seconds/60))*(seconds/60)
     for c in ['delta_G_manning','B_delta_route_length_mean','B_delta_area_proxy','path_delta_equivalent_length']:
      rounding.append(dict(method=method,child_state=state,response=response,variable=c,round_seconds=seconds,spearman=audit.corr(audit.clean(z[c]),yy,True)))
 pd.DataFrame(rounding).to_csv(O/'response_tie_sensitivity.csv',index=False);pd.DataFrame(counts).to_csv(O/'predictor_tolerance_sensitivity.csv',index=False)
 # Mechanistic signature comparisons for shortlist pairs; use only genuinely varying
 # type/descriptor combinations (exclude roundoff-only null-stratum signatures).
 ev=effects[effects.method.eq('variable_manning')].merge(cov[['variable','child_state','varying']],on=['variable','child_state']);ev=ev[ev.varying]
 sig=[]
 for a,b in itertools.combinations(SHORT,2):
  values=[]
  for state in resolved:
   t=ev[ev.child_state.eq(state)&ev.variable.isin([a,b])].pivot(index='term',columns='variable',values='variance_pct')
   if a not in t or b not in t:continue
   aa=t[a].to_numpy()/100;bb=t[b].to_numpy()/100
   if not np.isfinite(aa).all() or not np.isfinite(bb).all():continue
   aa=np.r_[aa,max(0,1-aa.sum())];bb=np.r_[bb,max(0,1-bb.sum())]
   values.append(.5*np.sum(abs(aa-bb)))
  pair=pairs[((pairs.a==a)&(pairs.b==b))|((pairs.a==b)&(pairs.b==a))]
  sig.append(dict(a=a,b=b,mean_signature_total_variation=np.mean(values) if values else np.nan,n_types=len(values),**(pair.drop(columns=['a','b']).iloc[0].to_dict() if len(pair) else {})))
 pd.DataFrame(sig).to_csv(O/'shortlist_pair_evidence.csv',index=False)
 # Figures: selection accounting, associations, redundancy, mechanisms and illustrative pairs.
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
 fig,ax=plt.subplots(figsize=(9,4));counts0=inv.status.value_counts().reindex(['within_type_varying','between_type_only','partly_defined','globally_constant']);names=['Varies within at least one type','Between-type variation only','Partly defined','Invariant across this grid']
 bars=ax.barh(names,counts0,color=['#2563eb','#d97706','#64748b','#94a3b8']);ax.bar_label(bars,padding=4);ax.invert_yaxis();ax.set(xlim=(0,155),xlabel='Candidate representations',title='205 representations: mathematical coverage and available variation');savefig(fig,'01_inventory')
 fig,axes=plt.subplots(1,2,figsize=(12,8),sharey=True,constrained_layout=True)
 for ax,response in zip(axes,audit.RESPONSES):
  q=summary[(summary.method=='variable_manning')&(summary.response==response)].set_index('variable').loc[SHORT]
  positions=np.arange(len(SHORT));ax.plot(q.mean_abs_spearman,positions,'o',label='Mean |Spearman|',color='#2563eb');ax.plot(q.mean_abs_pearson,positions,'x',label='Mean |Pearson|',color='#d97706')
  ax.hlines(positions,q.min_abs_spearman,q.max_abs_spearman,color='#2563eb',alpha=.3);ax.set(xlim=(0,1.03),yticks=positions,yticklabels=[LABEL[c]+(' [context]' if tier[c]=='context reserve' else ' [swap]' if tier[c]=='alternative' else '') for c in SHORT],xlabel='Absolute correlation',title='Centroid shift' if response.startswith('centroid') else 'Peak shift');ax.grid(axis='x',alpha=.15)
 axes[0].invert_yaxis();axes[1].legend(loc='lower right');fig.suptitle('Candidate pool · variable Manning · equal weighting of resolved types\nBars show between-type ranges, not confidence intervals; missing values are not zeros');savefig(fig,'02_response_associations')
 mat=pd.read_csv(O/'spearman_pooled.csv',index_col=0).loc[SHORT,SHORT]
 fig,ax=plt.subplots(figsize=(12,10),constrained_layout=True);im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=range(len(SHORT)),yticks=range(len(SHORT)),xticklabels=[LABEL[c] for c in SHORT],yticklabels=[LABEL[c] for c in SHORT],title='Predictor–predictor Spearman · all seven geometries pooled\nClustering additionally checks every within-type relationship');plt.setp(ax.get_xticklabels(),rotation=55,ha='right');fig.colorbar(im,ax=ax,label='Signed Spearman');savefig(fig,'03_predictor_correlation')
 selected=['B_delta_route_length_mean','delta_G_manning','B_delta_area_proxy','B_parent_route_length_sd','B_parent_dominance_mean','path_parent_signed_length_width_association','path_parent_slope_mean']
 mm=ev[ev.variable.isin(selected)&ev.child_state.isin(resolved)&ev.order.eq(1)].pivot_table(index='term',columns='variable',values='variance_pct',aggfunc='mean').reindex(index=audit.FACTORS,columns=selected)
 fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True);im=ax.imshow(mm,vmin=0,vmax=100,cmap='viridis',aspect='auto');ax.set(xticks=range(len(selected)),yticks=range(8),xticklabels=[LABEL[c] for c in selected],yticklabels=[c.replace('_',' ') for c in audit.FACTORS],title='Which controls change each descriptor? Main-effect variance shares\nMean over resolved types where the descriptor varies; interactions excluded here')
 for i in range(8):
  for j in range(len(selected)):
   v=mm.iloc[i,j]
   if np.isfinite(v):ax.text(j,i,('0' if v<1e-8 else '<0.1' if v<.05 else f'{v:.1f}'),ha='center',va='center',color='white' if v<45 else 'black',fontsize=9)
 plt.setp(ax.get_xticklabels(),rotation=40,ha='right');fig.colorbar(im,ax=ax,label='% of descriptor variance');savefig(fig,'04_mechanisms')
 ex=[('B_delta_route_length_mean','delta_G_manning'),('B_delta_route_length_mean','path_delta_equivalent_length'),('B_parent_route_length_sd','B_delta_route_length_sd'),('B_parent_dominance_mean','path_parent_signed_length_width_association')]
 fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
 for ax,(a,b) in zip(axes.flat,ex):
  for state,z in g.groupby('child_state'):
   z=z[[a,b]].drop_duplicates();ax.scatter(z[a],z[b],s=7,alpha=.35,label=state,rasterized=True)
  units={'B_delta_route_length_mean':' (m)','B_delta_route_length_sd':' (m)','B_parent_route_length_sd':' (m)','path_delta_equivalent_length':' (m)','delta_G_manning':r' ($m^{7/5}$)'}
  ax.set(xlabel=LABEL[a]+units.get(a,' (dimensionless)'),ylabel=LABEL[b]+units.get(b,' (dimensionless)'));ax.ticklabel_format(style='sci',axis='both',scilimits=(-3,4));ax.grid(alpha=.12)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8);fig.suptitle('High rank correlation is not a universal conversion\nUnique coordinate pairs plotted; no random sampling');savefig(fig,'05_pair_examples')
 sources={str(p):audit.sha(p) for p in [Path(__file__),Path(audit.__file__),O/'provenance.json',S/'factorial_signatures.csv',O/'inventory.csv',O/'predictor_pairs.csv',O/'response_associations.csv']}
 (O/'shortlist_provenance.json').write_text(json.dumps(dict(reference='SYN-PRE-RF-SHORTLIST-2026-10-30',sources=sources,n_shortlisted=len(SHORT),tiers=short.tier.value_counts().to_dict(),decision='Analyst-curated, exploratory shortlist informed by complete factorial evidence, mathematical coverage, interpretable mechanisms and cross-method checks; not an optimal subset or held-out selection.',rf_fits=0,routing_runs=0),indent=2))
 print('DECISIONS',ledger.decision.value_counts().to_dict());print('DONE')
if __name__=='__main__':main()
