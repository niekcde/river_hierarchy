"""Read-only preflight of refreshed real contexts; never modifies production runs."""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
import pandas as pd
import pyogrio
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'RAPID/src'))
from rapid_tools.reference_section import predict_based_depth
OUT=ROOT/'RAPID/assets/refreshed_hydraulic_stability_20261002'
DATA=ROOT/'network_variants/output'
EXAMPLES=['02','03','05','11','13','15','21','22','37','41','45','46','48','54','55','56']

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 refs=[];rows=[];skips=[];inventory=[]
 for ex in EXAMPLES:
  suffix='_direction130' if ex=='15' else ''
  base=DATA/f'{ex}_indep{suffix}__anchored'
  bm=base/'states/S000_base/hydraulic_context/hydraulic_context_manifest.json'
  m=json.loads(bm.read_text());ref=json.loads((base/'reference_section/reference_kb_summary.json').read_text())
  slope=float(m['section_slope_summary']['slope']);old_s=float(ref['reference_slope']);w=float(ref['reference_width_m']);q=float(ref['reference_discharge_cms']);old_d=float(ref['reference_depth_m'])
  with threadpool_limits(limits=1):new_d=predict_based_depth(discharge_cms=q,width_m=w,slope=slope)
  implied=lambda d,s:w*d**(5/3)*np.sqrt(s)/q
  refs.append(dict(example=ex,reference_width_m=w,reference_q_cms=q,stored_slope=old_s,refreshed_slope=slope,stored_depth_m=old_d,rebased_depth_m=new_d,n_old=implied(old_d,old_s),n_fixed_depth_new_slope=implied(old_d,slope),n_rebased_new_slope=implied(new_d,slope),base_generated_at=m['generated_at']))
  cfgpath=base/'rapid_prep_manifest.json'
  cfg=json.loads(cfgpath.read_text())['prep_config'] if cfgpath.exists() else {}
  lo=float(cfg.get('min_celerity_mps',.05));hi=float(cfg.get('max_celerity_mps',2.524))
  for family in ['indep','compound_bubbles']:
   experiment=DATA/f'{ex}_{family}{suffix}__anchored'
   for mp in sorted(experiment.glob('states/*/hydraulic_context/hydraulic_context_manifest.json')):
    cm=json.loads(mp.read_text());state=mp.parent.parent.name
    try:
     if cm.get('generated_at','')<m['generated_at']:
      raise ValueError('Context predates refreshed base')
     cp=mp.parent/'link_hydraulic_context.csv';ctx=pd.read_csv(cp,usecols=['id_link','link_length_m','slope_used'])
     wp=Path(cm['input_paths']['directed_links'])
     widths=pyogrio.read_dataframe(wp,columns=['id_link','wid_adj_routing'],read_geometry=False)
     frame=ctx.merge(widths,on='id_link',validate='one_to_one')
     if len(frame)!=len(ctx):raise ValueError('Unmatched link widths')
     a=frame[['link_length_m','slope_used','wid_adj_routing']].to_numpy(float)
     if not (np.isfinite(a).all() and (a>0).all()):raise ValueError('Invalid length/slope/width')
     length,s,wlocal=a.T
     count=np.maximum(1,np.rint(length/200)).astype(int)
     effective=np.maximum(200,length/count)
     inventory.append(dict(example=ex,family=family,state=state,n_links=len(frame),generated_at=cm['generated_at'],context_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),width_source=str(wp),stored_n=cfg.get('n_manning'),stored_alpha=1. if cfg else None,cap_source='saved_prep' if cfg else 'diagnostic_defaults',cap_min=lo,cap_max=hi))
     for anchor,depth in [('stored_BASED',old_d),('refreshed_BASED',new_d)]:
      for alpha in [1.,.71,.65,.6]:
       depths=depth*(wlocal/w)**alpha
       for label,n in [(str(n),n) for n in [.015,.025,.035,.05,.075]]+[('implied',implied(depth,slope))]:
        raw=(5/(3*n))*depths**(2/3)*np.sqrt(s)
        for caps in ['uncapped','configured_or_default_caps']:
         c=np.clip(raw,lo,hi) if caps=='configured_or_default_caps' else raw
         k=effective/c;lower=.2*k;upper=1.8*k
         il=int(lower.argmax());iu=int(upper.argmin());low=float(lower[il]);high=float(upper[iu])
         integer_low=math.ceil(low-1e-9);integer_high=math.floor(high+1e-9)
         rows.append(dict(example=ex,family=family,state=state,anchor=anchor,alpha=alpha,n_label=label,n=n,caps=caps,lower_seconds=low,upper_seconds=high,integer_feasible=integer_low<=integer_high,continuous_feasible=low<=high,margin_seconds=high-low,slow_link=int(frame.id_link.iloc[il]),fast_link=int(frame.id_link.iloc[iu]),slow_width_m=wlocal[il],slow_slope=s[il],slow_depth_m=depths[il],fast_width_m=wlocal[iu],fast_slope=s[iu],lower_cap_length_fraction=float(np.average(raw<lo,weights=length)),upper_cap_length_fraction=float(np.average(raw>hi,weights=length))))
    except (ValueError,KeyError,OSError) as err:
     skips.append(dict(example=ex,family=family,state=state,reason=str(err)))
  print(ex,'complete',len(inventory),'states',flush=True)
  pd.DataFrame(refs).to_csv(OUT/'references.csv',index=False)
 d=pd.DataFrame(rows);d.to_parquet(OUT/'stability.parquet',index=False)
 pd.DataFrame(inventory).to_csv(OUT/'inventory.csv',index=False)
 pd.DataFrame(skips,columns=['example','family','state','reason']).to_csv(OUT/'excluded.csv',index=False)
 for scope,g in [('all_states',d),('base',d[(d.family=='indep')&(d.state=='S000_base')])]:
  summary=g.groupby(['anchor','alpha','n_label','caps']).integer_feasible.agg(['size','sum']).reset_index();summary['failed']=summary['size']-summary['sum'];summary.to_csv(OUT/f'{scope}_summary.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps(dict(reference='REAL-HYDRAULIC-PREFLIGHT-2026-10-02',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_root=str(DATA.resolve()),scope='Independent and compound refreshed contexts; island refresh not assumed complete. Excludes contexts predating refreshed base.',settings='200m target using production round subdivision; 200m effective floor; X=.1; per-state integer dt interval. No adaptive effective-length caps, forcing-clock divisibility search or shared comparison clock.',anchors='Stored BASED depth and diagnostic BASED re-evaluation at refreshed base reference slope. Neither changes production references.',caveat='Nominal preflight not full preparation success or physical validation. Caps from independent prep where available; .05/2.524 diagnostic defaults for 13,45,48 without saved prep. Scope is a snapshot; 37 has unresolved equivalent-resistance solver failures. No rerouting.',n_states=len(inventory),n_excluded=len(skips)),indent=2))
 print('DONE',len(inventory),'states',len(skips),'excluded',len(d),'scenarios',flush=True)
if __name__=='__main__':main()
