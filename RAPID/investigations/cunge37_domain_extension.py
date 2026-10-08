"""Reference-state applicability screen across completed 37 states, no routing."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];SOURCE=Path('/Volumes/PhD/river_hierarchy/cunge_full_hierarchy_runs/37');OUT=ROOT/'RAPID/outputs/cunge_fixed_comparison_plus37_20261006'
r=pd.read_csv(SOURCE/'state_run_registry.csv').drop_duplicates('fingerprint');a=pd.read_csv(ROOT/'RAPID/outputs/cunge_domain_quick_check_20261006/summary.csv',dtype={'example':str});rise=float(a[a.example.eq('37')].iloc[0].rise_hours)*3600;rows=[]
for v in r.itertuples():
 f=pd.read_parquet(Path(v.output_dir)/'parameters.parquet');h=f.depth_m;S=f.slope;u=f.rapid_celerity_mps_raw*3/5;F2=u*u/(9.80665*h);Tplus=rise*u*S/h;D=rise*S*np.sqrt(9.80665/h);w=f.link_length_m
 old=(Tplus>=80)&(F2<=.9);new=(D>=30)&(F2<=.9)
 rows.append(dict(state=v.state,fingerprint=v.fingerprint,physical_links=len(f),old_screen_length_pct=100*w[old].sum()/w.sum(),diffusion_and_subcritical_length_pct=100*w[new].sum()/w.sum(),half_period_diffusion_and_subcritical_length_pct=100*w[(D*.5>=30)&(F2<=.9)].sum()/w.sum(),rise_proxy_h=rise/3600))
pd.DataFrame(rows).to_csv(OUT/'37_domain_screen.csv',index=False)
print(pd.DataFrame(rows).describe().to_string())
