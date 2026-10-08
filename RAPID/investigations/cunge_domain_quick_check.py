"""Reference-state wave-regime screening, no routing or calibration changes."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/outputs/cunge_domain_quick_check_20261006';OUT.mkdir(parents=True,exist_ok=True)
c=pd.read_csv(ROOT/'RAPID/outputs/cunge_calibration_alpha2/comparison.csv',dtype={'example':str});rows=[];sources=[]
for ex,z in c.groupby('example'):
 p=Path(z[z.method=='cunge'].iloc[0].output_dir);f=pd.read_parquet(p/'frozen_grid.parquet');w=pd.read_parquet(p/'cunge_best_wave.parquet');m=json.loads((p/'case.json').read_text());q=w.q_inflow_cms.to_numpy();t=np.arange(len(q))*m['forcing_step_s'];peak=int(q.argmax());threshold=q[0]+.01*(q.max()-q[0]);j=int(np.flatnonzero(q[:peak+1]>threshold)[0]);assert j>0
 onset=t[j-1]+(t[j]-t[j-1])*(threshold-q[j-1])/(q[j]-q[j-1]);rise=t[peak]-onset;assert rise>0
 sources.append(dict(example=ex,source=str(p),rise_hours=rise/3600,frame_sha256=hashlib.sha256((p/'frozen_grid.parquet').read_bytes()).hexdigest(),wave_sha256=hashlib.sha256((p/'cunge_best_wave.parquet').read_bytes()).hexdigest()))
 for setting,n in [('old_n_current_geometry',.035),('fixed_calibrated',float(z[z.method=='fixed_x'].iloc[0].n_manning)),('cunge',float(z[z.method=='cunge'].iloc[0].n_manning))]:
  for flow,Q in [('initial',q[0]),('peak_reference',m['reference_discharge_cms'])]:
   out=f[['id_link','link_length_m','width','slope','depth_m']].copy();out['example']=ex;out['setting']=setting;out['n']=n;out['flow_reference']=flow;out['rise_hours']=rise/3600
   um=f.depth_m**(2/3)*np.sqrt(f.slope)/n;qa=f.reference_flow_fraction*Q;ua=qa/(f.width*f.depth_m);out['manning_Q_over_allocated_Q']=um*f.width*f.depth_m/qa
   for name,u in [('manning',um),('allocated',ua)]:
    out[name+'_f2']=u*u/(9.80665*f.depth_m);out[name+'_tplus']=rise*u*f.slope/f.depth_m;out[name+'_old_screen']=(out[name+'_f2']<=.9)&(out[name+'_tplus']>=80)
   out['diffusion_index']=rise*f.slope*np.sqrt(9.80665/f.depth_m)
   out['diffusion_period_screen']=out.diffusion_index>=30
   # Retain the repository subcritical screen as an extra, explicit condition;
   # this is not represented as the complete Ponce/Moussa applicability region.
   for name in ['manning','allocated']:
    out[name+'_diffusion_and_subcritical']=out.diffusion_period_screen&(out[name+'_f2']<=.9)
   rows.append(out)
d=pd.concat(rows,ignore_index=True);d.to_parquet(OUT/'reference_coordinates.parquet',index=False);summ=[]
for (ex,setting,flow),g in d.groupby(['example','setting','flow_reference']):
 for mode in ['manning','allocated']:
  weight=g.link_length_m;den=weight.sum();summ.append(dict(example=ex,setting=setting,flow_reference=flow,velocity=mode,physical_links=len(g),rise_hours=g.rise_hours.iloc[0],old_screen_length_pct=100*weight[g[mode+'_old_screen']].sum()/den,diffusion_period_screen_length_pct=100*weight[g.diffusion_period_screen].sum()/den,diffusion_and_subcritical_length_pct=100*weight[g[mode+'_diffusion_and_subcritical']].sum()/den,diffusion_half_period_length_pct=100*weight[g.diffusion_index*.5>=30].sum()/den,diffusion_double_period_length_pct=100*weight[g.diffusion_index*2>=30].sum()/den,median_f2=g[mode+'_f2'].median(),median_tplus=g[mode+'_tplus'].median(),median_Qmanning_over_Qallocated=g.manning_Q_over_allocated_Q.median()))
s=pd.DataFrame(summ);s.to_csv(OUT/'summary.csv',index=False)
# Hydraulic period coordinate independent of n/X at fixed geometry and forcing.
for key,g in d.groupby(['example','id_link','flow_reference']):assert np.ptp(g.diffusion_index)==0 and np.ptp(g.allocated_f2)==0 and np.ptp(g.allocated_tplus)==0
(OUT/'provenance.json').write_text(json.dumps(dict(sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='16 base networks only, including excluded case11 for diagnostics. Same current geometry/forcing in every n comparison; not a historical rerun. Physical-length weighting, not subsection counting.',definitions='F2=u^2/(g*h), Tplus=T*u*S/h; u separately raw Manning and allocated Q/(B*h). Never capped celerity as velocity. T is first 1% departure-to-first-inlet-peak proxy, not measured perturbation period. S is stored slope, not a solved transient friction slope.',screens='Repository Tplus>=80 and F2<=0.9 versus diffusion timescale T*S*sqrt(g/h)>=30; optional F2<=0.9 applied separately. Half/double T are sensitivity, not alternative accepted periods. Neither is full model validation.',references=['https://ponce.sdsu.edu/when_is_the_diffusion_wave_applicable.html','https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-cunge-model','https://gmao.gsfc.nasa.gov/media/gmaoftp/sarith/ROUTING_MODEL/docs/amazon_wrcr20212.pdf']),indent=2))
a=s[(s.setting=='cunge')&(s.flow_reference=='peak_reference')&(s.velocity=='manning')];print(a[['example','rise_hours','old_screen_length_pct','diffusion_period_screen_length_pct','diffusion_and_subcritical_length_pct','diffusion_half_period_length_pct','diffusion_double_period_length_pct']].round(2).to_string(index=False))
b=s[(s.flow_reference=='peak_reference')&(s.velocity=='manning')];print(b.pivot(index='example',columns='setting',values='old_screen_length_pct').round(2).to_string())
report='''CUNGE HYDRAULIC DOMAIN QUICK CHECK — 6 October 2026

The notebook's Tplus>=80 / F2<=0.9 is explicitly a provisional screen, not the full physical applicability boundary. Diffusion-wave routing admits attenuation and has a broader applicability range than the pure kinematic approximation. This does not mean all fixed-X Muskingum is kinematic, or that all Cunge implementations are valid throughout every diffusion-wave situation.

Applied the published indicative diffusion-timescale criterion T*S*sqrt(g/h)>=30 to 16 existing base networks, using the current first-1%-departure-to-inlet-peak duration as a period proxy. Reused exact current physical widths, depths and slopes. Separately assessed raw Manning velocities at n=.035, the calibrated fixed-X n and calibrated Cunge n. The .035 comparison is a controlled roughness counterfactual under current geometry/forcing, not a historical state reconstruction. Allocated-Q velocities are also reported for initial and peak reference Q. No rerouting or calibration changes.

The diffusion index does not depend on n or X when T, depth and slope are fixed. The allocated-flow F2 and Tplus coordinates also do not depend on n or X. Changing n moves the Manning-implied coordinates: higher n lowers both F2 and Tplus. This may help one screen component while hurting the other. A better hydrograph or predictor fit does not by itself prove better wave-regime placement.

All percentages below are fractions of summed PHYSICAL link length, including parallel branches, not fractions of valid rivers or probability of validity. T is a proxy, not a measured sinusoidal perturbation period; half/double T demonstrate sensitivity. The extra subcritical condition retains the repository F2<=0.9 as an additional screen. Neither criterion tests backwater, inertial effects in detail, flow/depth/Manning consistency, time-varying geometry or numerical accuracy. Case11 appears only as a diagnostic, not reinstated as a usable Cunge calibration.

Current Cunge base-state screening (raw Manning velocity):
'''+a[['example','rise_hours','old_screen_length_pct','diffusion_period_screen_length_pct','diffusion_and_subcritical_length_pct','diffusion_half_period_length_pct','diffusion_double_period_length_pct']].round(3).to_string(index=False)+'\n\nOriginal provisional screen at fixed current geometry and event, different n:\n'+b.pivot(index='example',columns='setting',values='old_screen_length_pct').round(3).to_string()+'''

Interpretation: there is substantially more compatibility with an indicative diffusion-wave domain than our rectangular provisional screen suggests. That is a change in the applicability criterion, not evidence that changing X moved the river into that domain. Our constant-reference MC coefficients are not the full variable-depth/time-varying implementation described by HEC-HMS. Width-based branch allocation and BASED depths can still be inconsistent with Manning-implied discharge, as recorded in summary.csv; switching X does not repair that. Fixed-X empirical routing has no universal physical boundary that can simply be compared with this one. The matched geometric comparison domains used by our predictors remain the same.

Sources:
https://ponce.sdsu.edu/when_is_the_diffusion_wave_applicable.html (Ponce 2023, explaining Ponce et al. 1978 criteria)
https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-cunge-model (diffusion approximation and variable coefficients)
https://gmao.gsfc.nasa.gov/media/gmaoftp/sarith/ROUTING_MODEL/docs/amazon_wrcr20212.pdf (Getirana & Paiva 2013 wave-regime framework)
'''
(OUT/'REPORT.txt').write_text(report)
