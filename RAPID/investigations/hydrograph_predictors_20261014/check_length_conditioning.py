"""Check whether the reported within-network length association is independent.

Algebraic check, not a conditional-independence test across networks. No fits.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'RAPID/outputs/hydrograph_residual_review_20261018'

def main():
 source=OUT/'diagnostic_rows.parquet';d=pd.read_parquet(source);rows=[]
 for (scenario,ex),g in d.groupby(['scenario','example']):
  L=g.B_parent_route_length_mean.to_numpy();f=g.B_mean_length_fraction.to_numpy();N=g.N_parent_route_length_mean.to_numpy()
  assert np.ptp(N)<1e-8
  np.testing.assert_allclose(L,f*N,rtol=1e-12,atol=1e-8)
  rankdiff=np.max(abs(pd.Series(L).rank().to_numpy()-pd.Series(f).rank().to_numpy()))
  assert rankdiff==0
  rows.append(dict(scenario=scenario,example=ex,n=len(g),network_mean_length_m=N[0],
   max_identity_error_m=float(np.max(abs(L-f*N))),max_rank_difference=float(rankdiff),
   rho_absolute_length=float(spearmanr(L,g.error_min).statistic),
   rho_existing_relative_size=float(spearmanr(f,g.error_min).statistic)))
 a=pd.DataFrame(rows);a.to_csv(OUT/'parent_length_conditioning_check.csv',index=False)
 report=dict(identity='mu_L,parent,D = f_mu * mu_L,parent,N',
  conclusion='Within a network, conditioning on the existing f_mu leaves no independent absolute-domain-length variation. A partial association is undefined, not evidence of zero global predictive utility. Adding absolute length could still introduce between-network scale, which this check does not test.',
  scenarios=a.groupby('scenario').size().to_dict(),max_identity_error_m=float(a.max_identity_error_m.max()),
  max_correlation_difference=float(abs(a.rho_absolute_length-a.rho_existing_relative_size).max()),
  model_fits=0,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]})
 (OUT/'parent_length_conditioning_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 print(a[a.scenario.eq('all')].to_string(index=False))
if __name__=='__main__':main()
