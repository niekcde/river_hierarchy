"""Identify controlling and conflicting links under the selected hydraulic diagnostic."""
from pathlib import Path
import json
import numpy as np,pandas as pd,pyogrio
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
def main():
 refs=pd.read_csv(OUT/'hydraulic_references.csv',dtype={'example':str}).set_index('example');audit=pd.read_csv(OUT/'hydraulic_stability.csv',dtype={'example':str});failed=audit[(audit.anchor=='stored_BASED')&(audit.alpha==.65)&(audit.n_label=='0.035')&(audit.caps=='common_0.05_2.524_caps')&~audit.integer_feasible];allrows=[];extrema=[]
 for ex in failed.example:
  folder=ROOT/'network_variants/output'/f'{ex}_indep{"_direction130" if ex=="15" else ""}__anchored';m=json.loads((folder/'states/S000_base/hydraulic_context/hydraulic_context_manifest.json').read_text());widths=pyogrio.read_dataframe(m['input_paths']['directed_links'],columns=['id_link','wid_adj_routing'],read_geometry=False);links=pd.read_parquet(OUT/ex/'floor_0.05/old/links.parquet').merge(widths,on='id_link',validate='one_to_one');nodes=pd.read_parquet(OUT/ex/'floor_0.05/old/nodes.parquet').set_index('id_node');a=refs.loc[ex]
  links['depth_m']=a.stored_depth_m*(links.wid_adj_routing/a.reference_width_m)**.65;links['celerity_raw_mps']=5/(3*.035)*links.depth_m**(2/3)*np.sqrt(links.slope);links['celerity_mps']=links.celerity_raw_mps.clip(.05,2.524);links['effective_length_m']=np.maximum(200,links.actual_length_m/np.maximum(1,np.rint(links.actual_length_m/200)));links['K_seconds']=links.effective_length_m/links.celerity_mps;links['dt_lower_s']=.2*links.K_seconds;links['dt_upper_s']=1.8*links.K_seconds;low=links.dt_lower_s.max();high=links.dt_upper_s.min();links['example']=ex;links['too_slow_for_current_fastest']=links.dt_lower_s>high;links['too_fast_for_current_slowest']=links.dt_upper_s<low;links['controls_lower_bound']=np.isclose(links.dt_lower_s,low);links['controls_upper_bound']=np.isclose(links.dt_upper_s,high);links['node_us_reference_km']=links.id_us_node.map(nodes.reference_section_dist_out_m)/1000;links['node_ds_reference_km']=links.id_ds_node.map(nodes.reference_section_dist_out_m)/1000;allrows.extend(links.to_dict('records'))
  for role,ix in [('slow',links.dt_lower_s.idxmax()),('fast',links.dt_upper_s.idxmin())]:
   row=links.loc[ix].to_dict();row['role']=role;extrema.append(row)
 pd.DataFrame(allrows).to_csv(OUT/'stability_link_details.csv',index=False);pd.DataFrame(extrema).to_csv(OUT/'controlling_links_drilldown.csv',index=False)
 d=pd.DataFrame(allrows);print(d.groupby('example')[['too_slow_for_current_fastest','too_fast_for_current_slowest']].sum().to_string())
if __name__=='__main__':main()
