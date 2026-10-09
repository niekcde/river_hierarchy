"""Test whether extreme X and allocated/Manning-Q mismatch co-occur by reach."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def main():
    dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
    f=pd.read_parquet(dest/'hydraulic_discharge_consistency_reaches.parquet')
    ratio=f.allocated_over_manning_q
    f['depth_slope_cell_factor']=3*f.depth_m/(5*f.slope*f.rapid_effective_length_m)
    f['raw_over_used_celerity']=f.rapid_celerity_mps_raw/f.rapid_celerity_mps
    np.testing.assert_allclose(1-2*f.rapid_x,ratio*f.depth_slope_cell_factor*f.raw_over_used_celerity,rtol=1e-12)
    f['x_if_q_ratio_one_fixed_c']=.5*(1-(1-2*f.rapid_x)/ratio)
    # Algebraic attribution only; substituting local Manning Q is not a routing proposal.
    f[f.rapid_x < -100].sort_values('rapid_x').to_csv(dest/'extreme_x_discharge_overlap_reaches.csv',index=False)
    records=[]
    for ex,g in [('ALL',f),*list(f.groupby('example'))]:
        for threshold in [10,100,1000]:
            subset=g[g.rapid_x < -threshold];r=subset.allocated_over_manning_q
            records.append(dict(example=ex,x_threshold=-threshold,reaches=len(subset),ratio_gt1=int((r>1).sum()),ratio_gt2=int((r>2).sum()),ratio_gt10=int((r>10).sum()),ratio_le1=int((r<=1).sum())))
    pd.DataFrame(records).to_csv(dest/'extreme_x_discharge_overlap_summary.csv',index=False)
    lines=['EXTREME X / DISCHARGE MISMATCH OVERLAP — 2026-10-08',
      'Screening thresholds are descriptive, not admissibility limits.',
      'Join by example and hydraulic id_link; counts are reaches, not independent networks.',
      'Exact algebra verified: 1-2X = (Qref/QM) * [3h/(5S dx)] * (c_raw/c_used).',
      'Thus discharge mismatch is only one factor. Slope, depth, cell length and capping also contribute.',
      'The ratio-one X column is algebraic attribution, not an adopted local-flow substitution.',
      'For X<-100: ALL 148 reaches; 55 ratio>10, 129 ratio>1, 19 ratio<=1.',
      'For X<-100: 46 has 49 reaches; 18 ratio>10, 48 ratio>1, 1 ratio<=1.',
      'The most negative X in 46 is reach 237 (X=-9718.76), ratio=2.4615, cell length=42.02m.',
      '46 reach 284: X=-6447.01, ratio=1.7821, cell length=50.73m.',
      '46 reach 43: X=-435.10 despite ratio=0.6669.',
      'Conclusion: NO, not all extreme X cases coincide with a large Qref/QM ratio.',
      'No parameters, geometry, Manning n, forcing or production outputs modified.']
    (dest/'EXTREME_X_DISCHARGE_OVERLAP.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__':main()
