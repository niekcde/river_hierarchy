"""Read-only hydraulic and active-clock audit; isolated review artifacts."""
from pathlib import Path
import sys,json,hashlib
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/high_speed_audit_20261020'
SOURCES={}
def read(p):
 p=Path(p);SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 if p.suffix=='.json':return json.loads(p.read_text())
 if p.suffix=='.gpkg':return gpd.read_file(p)
 if p.suffix=='.csv':return pd.read_csv(p)
 return pd.read_parquet(p)

def main():
 jobs=read(ROOT/'RAPID/outputs/cap_investigation_20261020/jobs.json')
 targets=read(ROOT/'RAPID/outputs/bounded_beta_calibration_20261009/timing_targets.json')['records']
 comparison=read(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv')
 sections=[];summaries=[];clocks=[]
 for ex in ['22','46']:
  path=Path(next(j['parameters'] for j in jobs if j['example']==ex and j['state_id']=='S000_base'))
  f=read(path);base=path.parents[3]
  meta=read(base/'hydraulic_context/hydraulic_context_manifest.json')
  source=read(meta['provenance']['source_frame'])
  nodes=read(base/'hydraulic_context/node_hydraulic_context.csv').set_index('id_node')
  geo=read(base/'variant/widths/links_with_width_families.gpkg')
  for col in ['slope','head_us','head_ds','width','depth_m']:
   np.testing.assert_allclose(f.set_index('id_link')[col].sort_index(),source.set_index('id_link')[col].sort_index(),rtol=1e-10)
  np.testing.assert_allclose((f.head_us-f.head_ds)/f.link_length_m,f.slope,rtol=1e-10)
  np.testing.assert_allclose((5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)/f.n_manning,f.rapid_celerity_mps_raw,rtol=1e-10)
  f['length_km']=f.link_length_m/1000;f['head_drop_m']=f.head_us-f.head_ds
  f['slope_m_per_km']=f.slope*1000;f['implied_velocity_mps']=f.rapid_celerity_mps_raw*.6
  f['allocated_Q_over_Wh_mps']=f.reference_discharge_cms/(f.width*f.depth_m)
  f['Manning_Q_over_allocated_Q']=f.implied_velocity_mps/f.allocated_Q_over_Wh_mps
  f['diagnostic_normal_depth_m']=(f.reference_discharge_cms*f.n_manning/(f.width*np.sqrt(f.slope)))**.6
  f['diagnostic_normal_depth_celerity_mps']=(5/3)*f.diagnostic_normal_depth_m**(2/3)*np.sqrt(f.slope)/f.n_manning
  f['n_required_for_c5']=f.n_manning*f.rapid_celerity_mps_raw/5
  f['depth_width_coefficient']=f.depth_m/f.width**.65
  np.testing.assert_allclose(f.depth_width_coefficient,f.depth_width_coefficient.iloc[0])
  f['weighted_cap_delay_minutes']=f.reference_flow_fraction*f.link_length_m*(1/f.rapid_celerity_mps-1/np.maximum(f.rapid_celerity_mps_raw,f.celerity_min_mps))/60
  for suffix,key in [('us','u'),('ds','v')]:
   f['profile_distance_'+suffix+'_km']=f[key].map(nodes.node_reference_section_dist_out_m)/1000
   f['raw_wse_'+suffix]=f[key].map(nodes.node_wse_raw)
  f['raw_wse_drop_m']=f.raw_wse_us-f.raw_wse_ds
  f['profile_span_over_length']=abs(f.profile_distance_us_km-f.profile_distance_ds_km)/f.length_km
  f['example']=ex;f.to_csv(OUT/f'{ex}_all_links.csv',index=False)
  selected=f[f.rapid_celerity_mps_raw>5].copy();selected.to_csv(OUT/f'{ex}_capped_links.csv',index=False)
  summary=dict(example=ex,capped_links=len(selected),total_links=len(f),cap_delay_minutes=float(selected.weighted_cap_delay_minutes.sum()),max_celerity=float(f.rapid_celerity_mps_raw.max()),median_capped_Q_mismatch=float(selected.Manning_Q_over_allocated_Q.median()),max_capped_Q_mismatch=float(selected.Manning_Q_over_allocated_Q.max()),depth_width_coefficient=float(f.depth_width_coefficient.iloc[0]))
  summaries.append(summary)
  top=selected.nlargest(12,'rapid_celerity_mps_raw')
  influential=selected.nlargest(6,'weighted_cap_delay_minutes')
  influential.to_csv(OUT/f'{ex}_largest_cap_time_contributors.csv',index=False)
  sections.append(f'<h2>{ex}: hydraulic inputs</h2><p>Depth follows A W^0.65; it is prescribed, not measured at each link. Heads and slopes match the saved reconstruction. Q mismatch is a consistency diagnostic: Manning-implied discharge divided by the independently allocated reference discharge, assuming rectangular area W h. It is not an observed velocity comparison.</p>')
  cols=['id_link','length_km','rapid_celerity_mps_raw','width','depth_m','slope_m_per_km','head_drop_m','raw_wse_drop_m','reference_flow_fraction','Manning_Q_over_allocated_Q','weighted_cap_delay_minutes','profile_distance_us_km','profile_distance_ds_km']
  sections.append(top[cols].round(4).to_html(index=False))
  sections.append('<h3>Largest contributions to parent mean-route-time cap delay</h3><p>These are analytic r L (1/c_capped − 1/c_raw) contributions, not independently simulated centroid changes.</p>'+influential[cols].round(4).to_html(index=False))
  merged=geo[['id_link','geometry']].merge(f,on='id_link',validate='one_to_one').to_crs(4326)
  qc=geo[['id_link','wid_adj_wet','wid_adj_routing','routing_to_wet_width_ratio','width_qc_reason','width_correction_flag']]
  qc.merge(selected[['id_link']],on='id_link').to_csv(OUT/f'{ex}_capped_width_qc.csv',index=False)
  centers_all=merged.geometry.map(lambda g:g.interpolate(.5,normalized=True))
  pd.DataFrame(dict(id_link=merged.id_link,longitude=centers_all.map(lambda p:p.x),latitude=centers_all.map(lambda p:p.y))).to_csv(OUT/f'{ex}_link_locations.csv',index=False)
  fig=go.Figure()
  for label,mask,color in [('≤5 m/s',merged.rapid_celerity_mps_raw<=5,'#b8c0cc'),('5–8 m/s',merged.rapid_celerity_mps_raw.between(5,8,inclusive='right'),'#e9a23b'),('>8 m/s',merged.rapid_celerity_mps_raw>8,'#c63131')]:
   xs=[];ys=[];texts=[]
   for r in merged[mask].itertuples():
    for line in getattr(r.geometry,'geoms',[r.geometry]):
     xy=list(line.coords);xs.extend([z[0] for z in xy]+[None]);ys.extend([z[1] for z in xy]+[None]);texts.extend([f'Link {r.id_link}<br>c={r.rapid_celerity_mps_raw:.2f} m/s<br>L={r.length_km:.2f} km<br>S={r.slope_m_per_km:.3f} m/km<br>h={r.depth_m:.2f} m']*(len(xy)+1))
   fig.add_trace(go.Scatter(x=xs,y=ys,text=texts,hovertemplate='%{text}<extra></extra>',mode='lines',name=label,line=dict(color=color,width=2)))
  centers=merged[merged.id_link.isin(top.id_link)].copy();centers['pt']=centers.geometry.map(lambda g:g.interpolate(.5,normalized=True))
  fig.add_trace(go.Scatter(x=centers.pt.map(lambda p:p.x),y=centers.pt.map(lambda p:p.y),text=centers.id_link.astype(str),mode='markers+text',textposition='top center',name='Fastest links'))
  fig.update_layout(title=f'{ex}: geographic locations (longitude / latitude; no imagery)',height=650,template='plotly_white',xaxis_title='Longitude',yaxis_title='Latitude')
  sections.append(fig.to_html(full_html=False,include_plotlyjs=True if ex=='22' else False))
  profile=read(ROOT/f'RAPID/assets/bounded_robust_slopes_20261002/{ex}/floor_0.05/profile.parquet')
  bridges=read(ROOT/f'RAPID/assets/bounded_robust_slopes_20261002/{ex}/floor_0.05/manual_bridges.json')
  fig=go.Figure()
  for col in ['static_wse_m','fitted_wse_m']:
   fig.add_trace(go.Scatter(x=profile.reference_section_dist_out_m/1000,y=profile[col],name=col))
  for r in top.itertuples():
   fig.add_trace(go.Scatter(x=[r.profile_distance_ds_km,r.profile_distance_us_km],y=[r.head_ds,r.head_us],mode='lines+markers',name=f'link {r.id_link}'))
  for b in bridges:fig.add_vrect(x0=b['resolved_start_distance_m']/1000,x1=b['resolved_end_distance_m']/1000,fillcolor='orange',opacity=.2,line_width=0)
  fig.update_layout(title=f'{ex}: fitted reference profile and fastest link heads; shaded manual bridge',height=550,template='plotly_white',xaxis_title='Reference distance upstream from outlet (km)',yaxis_title='Elevation (m)')
  sections.append(fig.to_html(full_html=False,include_plotlyjs=False))
  row=comparison[(comparison.example.astype(str).str.zfill(2)==ex)&comparison.method.eq('cunge')].iloc[0]
  obs=read(Path(row.output_dir)/'observed_upstream.parquet')
  old=read(Path(row.output_dir)/'cunge_best_wave.parquet')
  current=read(ROOT/f'RAPID/outputs/common_grid_validation_20261008/a5520880422b/{ex}/60s.parquet')
  target=next(r for r in targets if r['example']==ex)
  fit=read(ROOT/f'RAPID/outputs/bounded_beta_calibration_20261009/{ex}/result.json')
  times=[pd.Timestamp(obs.loc[obs.discharge_cms.idxmax(),'time']),pd.Timestamp(old.loc[old.q_inflow_cms.idxmax(),'time_utc']),pd.Timestamp(current.loc[current.forcing_cms.idxmax(),'time_utc'])]
  L=float((f.reference_flow_fraction*f.link_length_m).sum())
  lag=(pd.Timestamp(fit['feature_utc'])-times[2]).total_seconds()
  clocks.append(dict(example=ex,observed_peak=str(times[0]),old_prepared_peak=str(times[1]),active_prepared_peak=str(times[2]),active_minus_observed_minutes=(times[2]-times[0]).total_seconds()/60,old_minus_observed_minutes=(times[1]-times[0]).total_seconds()/60,model_lag_hours=lag/3600,model_effective_celerity=L/lag,calibration_warnings=fit.get('warnings')))
  fig=go.Figure()
  for label,time,q in [('Observed upstream',pd.to_datetime(obs.time,utc=True),obs.discharge_cms),('Historical forcing',pd.to_datetime(old.time_utc,utc=True),old.q_inflow_cms),('Active calibration forcing',pd.to_datetime(current.time_utc,utc=True),current.forcing_cms)]:
   mask=(time>=times[0]-pd.Timedelta(hours=12))&(time<=times[0]+pd.Timedelta(hours=12))
   fig.add_trace(go.Scatter(x=time[mask],y=q[mask],name=label,mode='lines'))
  fig.update_layout(title=f'{ex}: inlet crest comparison (no timing shifts)',height=500,template='plotly_white',yaxis_title='Discharge (m³/s)')
  sections.append(fig.to_html(full_html=False,include_plotlyjs=False))
 pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
 (OUT/'clock_audit.json').write_text(json.dumps(clocks,indent=2))
 sections.insert(0,'<h1>High-speed hydraulic audit: 22 and 46</h1><p>Read-only investigation. No calibration, routing result or geometry changed. Maps show model geometry, not independent confirmation of rapids.</p>'+pd.DataFrame(clocks).to_html(index=False))
 evidence=read(ROOT/'RAPID/outputs/temporal_policy_review_20261008/mrc_clock_evidence.json')
 findings="""<h2>Findings and interpretation</h2>
 <p><b>Clock correction:</b> the previously quoted 308-minute discrepancy for 46 was historical. Current calibration forcing differs from the raw observed peak by only +30 seconds; for 22 it differs by −30 seconds. Current effective model crest speeds using probability-weighted network length are 1.604 m/s (46) and 3.364 m/s (22).</p>
 <p>For Pakse and Kratie, archived source CSV evidence shows the same conflict: headers say UTC+07:00 while timestamp strings end in Z. The actual timezone remains unconfirmed. If both stations use the same convention, a common seven-hour correction cancels in their relative lag. No timestamps were shifted in this audit.</p>
 <p><b>46:</b> links 292/307 lie in the Vermilion Chutes vicinity, and 22/55 around Boyer Rapids and its approach. The official Boyer Rapids position is 59.1761111 N, 112.6744444 W, close to link22's midpoint. These are spatial interpretations supported by named geographic features, not measured speeds. Raw endpoint WSE drops and reconstructed drops agree closely for all four. Their reference distances (17–25 and 234–240 km) do not overlap the manual 370–410 km bridge.</p>
 <p><b>22:</b> 33 of the 300 capped links have identical mapped reference-profile distances at both endpoints. Their reconstructed positive head drops therefore are not independently resolved by those profile locations. Fast examples321/172 are 248/225 m long and 2746/2150 m wide, with approximately 0.11/0.10 m reconstructed drops. Existing width QC flags both as inconsistent with the measured corridor; routing width exceeds wet width by about17%. These flags are evidence for closer review, not proof that every capped link is wrong.</p>
 <p><b>Hydraulic consistency:</b> assuming the rectangular area used in this diagnostic, prescribed depth and width combined with slope and calibrated n imply discharges 12–29 times allocated reference discharge for 46's four capped links. The capped-link median is3.52 in22. Reference inlet discharge is approximately the event peak, so this cannot simply be dismissed as a low-flow reference. Depth is exactly A W^0.65, not an independent local depth observation. Width, depth and branch flow allocation are therefore not a mutually solved normal-flow state. This limits literal physical interpretation of c=(5/3)h^(2/3)sqrt(S)/n.</p>
 <p>The normal-depth columns are algebraic consistency counterfactuals only; they are not measured depths, routed sensitivity runs or proposed replacements. Real rapids also need not satisfy uniform-flow assumptions. The audit does not establish which input should be corrected.</p>
 <p><b>Extremes versus impact:</b> the33 capped links with zero mapped profile span in22 contribute only2.17minutes (1.75%) of the parent's124.45-minute analytic cap delay. They do not explain the overall sensitivity. The largest individual contribution comes from link977:8.23km at7.69m/s, contributing9.60minutes. Links24,649,934,49,46 also deserve review. These analytic contributions are not simulated centroid effects.</p>
 <p><b>Decision:</b> genuine steep sections support investigating uncapped routing, but neither geography nor calibrated whole-network timing validates the local maxima. Keep existing results frozen. Prioritize22's long influential reaches alongside its short wide reconstructed extremes and the consistency of prescribed depth/allocated discharge; retain46 as a known-rapids sensitivity case. No independent-event validation was performed.</p>
 <p>External evidence:
 <a href="https://iaac-aeic.gc.ca/050/documents_staticpost/63919/85328/Vol2_Appendix_G.pdf">Site C technical report</a> describes roughly10m drop over3km at Vermilion Chutes;
 <a href="https://geogratis.gc.ca/services/geoname/en/geonames.html?category=O%2CP%2CM&amp;expand=items.status%2Citems.concise%2Citems.generic%2Citems.province&amp;q=&amp;select=items.status.term%2Citems.concise.term%2Citems.generic.term%2Citems.province.description&amp;skip=31000&amp;sort-field=name">Canadian Geographical Names Database</a> gives Boyer Rapids coordinates;
 <a href="https://publications.gc.ca/collections/collection_2025/eccc/en37/En37-601-1992-eng.pdf">Environment Canada sampling-reach report</a> describes bedrock rapids upstream of Peace Point. These sources establish physical features, not13m/s celerities.</p>"""
 sections.insert(1,findings)
 (OUT/'review.html').write_text('<html><meta charset="utf-8"><body style="font:15px Arial;margin:24px">'+''.join(sections)+'</body></html>')
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in SOURCES.items())
 (OUT/'manifest.json').write_text(json.dumps(dict(sources=SOURCES,source_hashes_unchanged=True),indent=2))
 print(pd.DataFrame(summaries).to_string(index=False));print(json.dumps(clocks,indent=2))
if __name__=='__main__':main()
