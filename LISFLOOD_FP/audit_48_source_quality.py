"""Trace 48 geometry flags back to stored source/QC records; read-only."""
from pathlib import Path
import json
import numpy as np,pandas as pd,pyogrio
from scipy.stats import theilslopes
import broad_study as b
r=b.OUT/'investigation';out=r/'geometry_context48';records=[]
for ex in sorted(b.CASES):
 f=pd.read_parquet(r/'current_mc/states'/ex/'S000_base/parameters.parquet');gid=f.geometry_id.iloc[0];meta=json.loads((b.ROOT/'RAPID/assets/master_integration_20261009/objects'/gid/'staged.json').read_text());path=Path(meta['paths']['links']);g=pyogrio.read_dataframe(path,read_geometry=False);g=g.set_index('id_link').loc[f.id_link].reset_index();assert np.allclose(g.wid_adj_routing,f.width,rtol=0,atol=1e-12)
 corrected=g.width_correction_applied.astype(bool);manualgood=g.manual_width_review_decision.fillna('').eq('good');flags=g.width_correction_flag.fillna(False).astype(bool);records.append(dict(example=ex,corrected_reaches=int(sum(corrected)),flagged_reaches=int(sum(flags)),manual_good_but_corrected=int(sum(manualgood&corrected)),minimum_correction_scale=float(g.width_correction_scale_factor.min())))
 if ex=='48':
  cols=['id_link','wid_adj_wet','wid_adj_total','wid_adj_routing','width_correction_applied','width_correction_scale_factor','width_qc_decision','width_qc_reason','width_trim_valid_core','width_trim_fallback_reason','invalid_sample_fraction','routing_corridor_consistency_flag','manual_width_review_decision','manual_width_review_requires_correction','width_correction_source'];g[cols].to_parquet(out/'48_width_source_qc.parquet',index=False)
  nodes=pd.read_csv(Path(meta['paths']['manifest']).parent/'node_hydraulic_context.csv');nodes.to_parquet(out/'48_node_source_snapshot.parquet',index=False);good=nodes.node_wse_raw.notna()&nodes.node_reference_section_dist_out_m.notna();x=nodes.loc[good,'node_reference_section_dist_out_m'];raw=nodes.loc[good,'node_wse_raw'];used=nodes.loc[good,'rapid_wse_for_slope'];ni=nodes.set_index('id_node');detail=[]
  for z in f[f.id_link.isin([36,38,45,65,130,161,163,164,165,188])].itertuples():
   u=ni.loc[z.u];v=ni.loc[z.v];detail.append(dict(id_link=z.id_link,length_m=z.link_length_m,width_m=z.width,used_slope_m_per_km=z.slope*1000,raw_endpoint_slope_m_per_km=(u.node_wse_raw-v.node_wse_raw)/z.link_length_m*1000,upstream_match_distance_m=u.sword_match_distance,downstream_match_distance_m=v.sword_match_distance))
  source=dict(raw_profile_ols_slope_m_per_km=float(np.polyfit(x,raw,1)[0]*1000),used_profile_ols_slope_m_per_km=float(np.polyfit(x,used,1)[0]*1000),raw_profile_theilsen_slope_m_per_km=float(theilslopes(raw,x).slope*1000),head_adjustment_median_m=float(np.median(abs(raw-used))),head_adjustment_max_m=float(max(abs(raw-used))),outlet=nodes[nodes.is_outlet][['id_node','node_wse_raw','rapid_wse_for_slope','sword_match_distance','sword_match_within_tolerance']].to_dict('records'),critical_reaches=detail,junction110=f[(f.u==110)|(f.v==110)][['id_link','u','v','width','reference_flow_fraction']].to_dict('records'))
  b.save(out/'48_source_diagnosis.json',source)
b.save(out/'width_qc_all_examples.json',records);print(pd.DataFrame(records).to_string(index=False))
# Equal roughness screen uses fixed current-MC discharge allocation; not a routed solution.
f=pd.read_parquet(out/'reach_comparison.parquet');screen=[]
for ex,g in f.groupby('example'):
 L=g.length_m.to_numpy();Fr=g.common_n_normal_fr.to_numpy();screen.append(dict(example=ex,n=.020799605931143154,length_weighted_mean_Fr=float(np.average(Fr,weights=L)),length_share_normal_Fr_gt06=float(sum(L[Fr>.6])/sum(L))))
b.save(out/'common_roughness_screen.json',screen)
# Exact saved starting state of the failed 50 m diagnostic.
d=dict(np.load(r/'limiter_retained_48_n0.0207996059_dx50/_input_29515.npz'));N=len(d['h']);eta=d['z'][:N]+d['h'];j=np.where(d['b']<N)[0];hf=np.maximum(np.maximum(eta[d['a'][j]],eta[d['b'][j]])-np.maximum(d['z'][d['a'][j]],d['z'][d['b'][j]]),1e-12);fr=abs(d['q'][j])/(hf*np.sqrt(9.80665*hf));cellfr=np.zeros(N);np.maximum.at(cellfr,d['a'][j],fr);b.save(out/'48_failed_run_initial_regime.json',dict(n=float(d['n']),grid='max50',length_fraction_outgoing_face_Fr_gt06=float(sum(d['dx'][cellfr>.6])/sum(d['dx'])),cell_depth_median_m=float(np.median(d['h'])),interpretation='Each cell is assigned the maximum internal outgoing-face Fr; length weighted. Screening statistic, not a universal stability criterion.'))
