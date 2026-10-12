"""Regenerate reviewed geometry attributes, using the existing domain-based slope implementation."""
from pathlib import Path
import sys,json,argparse,traceback
from functools import lru_cache
import numpy as np
import pandas as pd
import geopandas as gpd
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/src')]
from network_variants.reviewed_geometry import active_geometry,digest
from network_variants.shared_head_reconstruction import reconstruct_shared_heads
BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
OUT=ROOT/'RAPID/assets/regenerated_hydraulics_20261009'

@lru_cache(maxsize=16)
def base(ex):
 nodes=pd.read_parquet(BASE/ex/'floor_0.05/old/nodes.parquet')
 links=pd.read_parquet(BASE/ex/'floor_0.05/old/links.parquet')
 refs=pd.read_csv(BASE/'hydraulic_references.csv',dtype={'example':str}).set_index('example').loc[ex]
 return links,nodes,refs

def save(p,value):p.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')

def heads(r,release):
 folder=OUT/'objects'/r['geometry_id'];folder.mkdir(parents=True,exist_ok=True)
 links,nodes,contract,_,_=release.load(r['example'],r['family'],r['state'])
 bl,bn,refs=base(r['example']);reference=bn.set_index('id_node')['head']
 f=pd.DataFrame(links[['id_link','id_us_node','id_ds_node','u','v']]);f['link_length_m']=links.length.to_numpy()
 # Use the same domain-based solve as production, without extra fixed nodes.
 prior=folder/'head_result.json'
 if prior.exists():
  old=json.loads(prior.read_text())
  if old.get('status') in ['fixed_exterior_pass','blocked_fixed_exterior']:
   save(folder/'superseded_fixed_exterior_diagnostic.json',old)
 result,qa=reconstruct_shared_heads(f,bl,bn,reference)
 result.to_parquet(folder/'heads.parquet',index=False)
 report=dict(example=r['example'],family=r['family'],state=r['state'],geometry_id=r['geometry_id'],geometry_release=release.release,
             status='current_slope_pass',head_policy='existing_domain_policy',head_qa=qa,routing_ready=False)
 save(folder/'head_result.json',report);return report

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['heads','widths'],default='heads');parser.add_argument('--states',nargs='*');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 release=active_geometry();seen=set();records=[]
 for r in release.records.values():
  r=release.canonical(r)
  if not r.get('bundle') or r['geometry_id'] in seen:continue
  seen.add(r['geometry_id'])
  if args.states and r['state'] not in args.states:continue
  folder=OUT/'objects'/r['geometry_id'];folder.mkdir(parents=True,exist_ok=True)
  path=folder/('head_result.json' if args.stage=='heads' else 'width_result.json')
  if args.resume and path.exists():report=json.loads(path.read_text())
  else:
   try:
    if args.stage=='heads':report=heads(r,release)
    else:
     from regenerate_reviewed_widths import widths
     report=widths(r,release,folder)
   except Exception as error:
    report=dict(example=r['example'],state=r['state'],geometry_id=r['geometry_id'],status='error',reason=str(error),routing_ready=False)
    (folder/f'{args.stage}_error.txt').write_text(traceback.format_exc());save(path,report)
  records.append(report)
  if len(records)%25==0 or args.states:print(args.stage,len(records),r['example'],r['state'],report['status'],report.get('reason','')[:160],flush=True)
  if len(records)%25==0:pd.DataFrame(records).to_csv(OUT/f'{args.stage}_status.csv',index=False)
 name=f'{args.stage}_status'+('_selected' if args.states else '')
 pd.DataFrame(records).to_csv(OUT/f'{name}.csv',index=False)
 print(pd.DataFrame(records).status.value_counts().to_dict(),flush=True)
if __name__=='__main__':main()
