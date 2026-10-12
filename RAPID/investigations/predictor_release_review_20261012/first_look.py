"""Read-only first look at published October 2026 Cunge results; no model fitting."""
from pathlib import Path
from datetime import datetime
import pandas as pd,numpy as np,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs'/('predictor_release_review_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
OUT.mkdir(exist_ok=False);hashes={}
def read(p):
 p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return pd.read_csv(p,dtype={'example':str,'example_id':str}) if p.suffix=='.csv' else json.loads(p.read_text())
summary=read(ROOT/'RAPID/outputs/final_batch_review_20261011/summary.csv');verification=read(ROOT/'RAPID/outputs/final_batch_review_20261011/verification.json')
parts=[];rows=[];audit=[]
for ex in summary.example:
 registries=[]
 for fam in ['indep','compound_bubbles','island']:
  p=ROOT/f'network_variants/output/{ex}_{fam}__anchored/rapid_run_registry.csv'
  if not p.exists():continue # Some networks have no separately published island family.
  d=read(p);assert d.method.eq('cunge').all() and d.status.eq('ran').all();d['family']=fam;registries.append(d)
 d=pd.concat(registries,ignore_index=True)
 # Canonical physics identity; labels/families may be aliases.
 for fp,g in d.groupby('fingerprint'):
  assert g.outlet_parquet.nunique()==1 and np.ptp(g.peak_discharge_cms)<1e-8
 u=d.drop_duplicates('fingerprint').copy();b=u[u.state_id.eq('S000_base')];assert len(b)==1;b=b.iloc[0]
 assert len(u)==int(summary.set_index('example').loc[ex,'network_states'])
 base_feature=pd.Timestamp(b.calibration_feature_utc)
 all_family={fp:','.join(sorted(g.family.unique())) for fp,g in d.groupby('fingerprint')}
 count={fp:len(g) for fp,g in d.groupby('fingerprint')}
 for r in u.itertuples():
  result=read(r.result_json);audit.append(dict(example=ex,fingerprint=r.fingerprint,tail_settled=result.get('tail_settled'),min_flow=result.get('minimum_native_flow_cms'),feature_kind=result.get('feature_kind')))
  if r.fingerprint==b.fingerprint:continue
  rows.append(dict(example=ex,fingerprint=r.fingerprint,state_id=r.state_id,selection_families=all_family[r.fingerprint],registry_occurrences=count[r.fingerprint],feature_shift_minutes=(pd.Timestamp(r.calibration_feature_utc)-base_feature).total_seconds()/60,sampled_peak_shift_minutes=(r.peak_time_seconds-b.peak_time_seconds)/60,peak_total_change_pct=100*(r.peak_discharge_cms/b.peak_discharge_cms-1),base_feature_kind=b.calibration_feature_kind,child_feature_kind=r.calibration_feature_kind,forcing_step_s=r.forcing_dt_seconds,outlet_parquet=r.outlet_parquet,parameters_parquet=r.parameters_parquet))
 parts.append(dict(example=ex,registry_rows=len(d),unique_states=len(u),unique_children=len(u)-1,step_min=d.forcing_dt_seconds.min(),step_max=d.forcing_dt_seconds.max(),base_feature_kind=b.calibration_feature_kind))
pairs=pd.DataFrame(rows);a=pd.DataFrame(audit);coverage=pd.DataFrame(parts)
assert len(a)==2187 and len(pairs)==2172 and a.tail_settled.all() and (a.min_flow>=-1e-9).all()
print(coverage.to_string(index=False))
res=pairs.assign(positive=pairs.sampled_peak_shift_minutes>0,negative=pairs.sampled_peak_shift_minutes<0,zero=pairs.sampled_peak_shift_minutes==0,attenuated=pairs.peak_total_change_pct<0,positive_attenuated=(pairs.sampled_peak_shift_minutes>0)&(pairs.peak_total_change_pct<0)).groupby('example').agg(pairs=('state_id','size'),positive=('positive','sum'),negative=('negative','sum'),zero=('zero','sum'),min_peak_shift_min=('sampled_peak_shift_minutes','min'),max_peak_shift_min=('sampled_peak_shift_minutes','max'),attenuated=('attenuated','sum'),positive_attenuated=('positive_attenuated','sum'),min_peak_change_pct=('peak_total_change_pct','min'),max_peak_change_pct=('peak_total_change_pct','max'))
print(res.to_string());print('feature/peak shift disagreements',int((abs(pairs.feature_shift_minutes-pairs.sampled_peak_shift_minutes)>1e-8).sum()));print('feature kinds',pairs[['base_feature_kind','child_feature_kind']].value_counts().to_string())
pairs.to_csv(OUT/'canonical_child_first_look.csv',index=False);coverage.to_csv(OUT/'coverage.csv',index=False);res.to_csv(OUT/'response_summary.csv');a.to_csv(OUT/'completion_audit.csv',index=False)
assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items())
(OUT/'provenance.json').write_text(json.dumps(dict(source_hashes=hashes,source_unchanged=True,unique_network_states=len(a),unique_children=len(pairs),reference_sections=15,scope='Published registries and result metadata only. Sampled peaks and total-discharge changes, not historical interpolated lag or peak-excess targets. No rerouting, calibration, feature rebuild, fitting, or production writes.'),indent=2))
print('OUT',OUT)
