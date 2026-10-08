"""Base-only hydraulic preflight for isolated robust profiles; no routing/promotion."""
from pathlib import Path
import sys,json,math
import numpy as np,pandas as pd,pyogrio
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'RAPID/src'),str(ROOT)]
from rapid_tools.reference_section import predict_based_depth
OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002';DATA=ROOT/'network_variants/output'

def main():
 rows=[];references=[];limits=[]
 for dest in sorted(OUT.glob('*/floor_0.05')):
  ex=dest.parent.name;suffix='_direction130' if ex=='15' else '';base=DATA/f'{ex}_indep{suffix}__anchored';hc=base/'states/S000_base/hydraulic_context';manifest=json.loads((hc/'hydraulic_context_manifest.json').read_text());ref=json.loads((base/'reference_section/reference_kb_summary.json').read_text());prep=base/'rapid_prep_manifest.json';config=json.loads(prep.read_text())['prep_config'] if prep.exists() else {};lo=float(config.get('min_celerity_mps',.05));hi=float(config.get('max_celerity_mps',2.524));selected=json.loads((dest/'selection.json').read_text());nodes=pd.read_parquet(dest/'old/nodes.parquet');span=np.ptp(nodes.reference_section_dist_out_m);sref=selected['new_drop']/span;wref=float(ref['reference_width_m']);q=float(ref['reference_discharge_cms']);stored=float(ref['reference_depth_m']);updated=float(predict_based_depth(discharge_cms=q,width_m=wref,slope=sref));widths=pyogrio.read_dataframe(manifest['input_paths']['directed_links'],columns=['id_link','wid_adj_routing'],read_geometry=False);links=pd.read_parquet(dest/'old/links.parquet').merge(widths,on='id_link',validate='one_to_one');assert len(links)==selected['n_links'];length=links.actual_length_m.to_numpy();s=links.slope.to_numpy();w=links.wid_adj_routing.to_numpy();assert np.all(np.isfinite(w)&(w>0));count=np.maximum(1,np.rint(length/200)).astype(int);effective=np.maximum(200,length/count)
  references.append(dict(example=ex,new_global_slope=sref,reference_width_m=wref,reference_q_cms=q,stored_depth_m=stored,updated_depth_m=updated,n_implied_stored_depth=wref*stored**(5/3)*np.sqrt(sref)/q,n_implied_updated_depth=wref*updated**(5/3)*np.sqrt(sref)/q,cap_min=lo,cap_max=hi,cap_source='saved_prep' if prep.exists() else 'diagnostic_defaults'))
  for anchor,dref in [('stored_BASED',stored),('updated_BASED_diagnostic',updated)]:
   for alpha in [.6,.65,.71,1.]:
    depth=dref*(w/wref)**alpha
    for label,n in [('0.025',.025),('0.035',.035),('0.05',.05),('implied',wref*dref**(5/3)*np.sqrt(sref)/q)]:
     raw=5/(3*n)*depth**(2/3)*np.sqrt(s)
     for caps in ['uncapped','configured_or_default_caps','common_0.05_2.524_caps']:
      c=raw if caps=='uncapped' else np.clip(raw,.05,2.524) if caps=='common_0.05_2.524_caps' else np.clip(raw,lo,hi);cap_lo,cap_hi=(.05,2.524) if caps=='common_0.05_2.524_caps' else (lo,hi);k=effective/c;lower=.2*k;upper=1.8*k;il=int(lower.argmax());iu=int(upper.argmin());low=float(lower[il]);high=float(upper[iu]);rows.append(dict(example=ex,anchor=anchor,alpha=alpha,n_label=label,n=n,caps=caps,lower_seconds=low,upper_seconds=high,integer_feasible=math.ceil(low-1e-9)<=math.floor(high+1e-9),continuous_feasible=low<=high,margin_seconds=high-low,slow_link=int(links.id_link.iloc[il]),fast_link=int(links.id_link.iloc[iu]),slow_slope=s[il],slow_width_m=w[il],slow_depth_m=depth[il],fast_slope=s[iu],fast_width_m=w[iu],fast_depth_m=depth[iu],lower_cap_length_fraction=float(np.average(raw<cap_lo,weights=length)),upper_cap_length_fraction=float(np.average(raw>cap_hi,weights=length))))
  print(ex,'stability complete',flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'hydraulic_stability.csv',index=False);pd.DataFrame(references).to_csv(OUT/'hydraulic_references.csv',index=False);a=d.groupby(['anchor','alpha','n_label','caps']).integer_feasible.agg(['size','sum']).reset_index();a['failed']=a['size']-a['sum'];a.to_csv(OUT/'hydraulic_summary.csv',index=False)
 (OUT/'hydraulic_method.json').write_text(json.dumps(dict(scope='16 base networks only, no routing',depth='D = Dref*(W/Wref)^alpha; stored BASED depth primary, BASED reevaluated at candidate whole-reference slope as diagnostic',celerity='(5/(3n))*D^(2/3)*sqrt(S); configured/default caps audited separately',stability='200m target; production round subdivision and 200m effective-length floor; X=.1; common integer dt must lie in intersection [0.2K,1.8K]',limits='Necessary coefficient-positivity preflight, not full preparation/forcing-clock success; no collapse states recomputed; no adaptive caps; implied n uses wide rectangular Manning diagnostic'),indent=2))
if __name__=='__main__':main()
