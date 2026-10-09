"""Compare allocated reference Q with the Manning state used for Cunge celerity.
Diagnostic only: no reference-flow, geometry, forcing or parameter changes.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def main():
    dest=Path(json.loads((ROOT/'RAPID/outputs/common_grid_validation_20261008/latest.json').read_text())['directory'])
    source=dest/'x_audit_reaches.parquet'
    f=pd.read_parquet(source)
    f['manning_discharge_cms']=f.width/f.n_manning*f.depth_m**(5/3)*np.sqrt(f.slope)
    np.testing.assert_allclose(f.manning_discharge_cms,.6*f.rapid_celerity_mps_raw*f.width*f.depth_m)
    f['allocated_over_manning_q']=f.reference_discharge_cms/f.manning_discharge_cms
    f['depth_required_for_allocated_q_m']=(f.reference_discharge_cms*f.n_manning/(f.width*np.sqrt(f.slope)))**(.6)
    f['required_over_prescribed_depth']=f.depth_required_for_allocated_q_m/f.depth_m
    reconstructed=f.width/f.n_manning*f.depth_required_for_allocated_q_m**(5/3)*np.sqrt(f.slope)
    np.testing.assert_allclose(reconstructed,f.reference_discharge_cms)
    # Deliberately DO NOT replace routed Q with this diagnostic local capacity:
    # those independently inferred discharges need not conserve network flow.
    f.to_parquet(dest/'hydraulic_discharge_consistency_reaches.parquet',index=False)
    records=[]
    for ex,g in f.groupby('example'):
        ratio=g.allocated_over_manning_q
        records.append(dict(example=ex,links=len(g),ratio_min=ratio.min(),ratio_median=ratio.median(),ratio_p95=ratio.quantile(.95),ratio_max=ratio.max(),links_ratio_gt10=int((ratio>10).sum()),links_ratio_lt01=int((ratio<.1).sum()),median_required_depth_factor=g.required_over_prescribed_depth.median()))
    out=pd.DataFrame(records)
    out.to_csv(dest/'hydraulic_discharge_consistency_summary.csv',index=False)
    (dest/'hydraulic_discharge_consistency_protocol.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),assumption='Wide rectangular channel; same Manning approximation as raw celerity. Uncapped c used for algebraic comparison.',n_policy='Existing per-example fitted Cunge n held fixed; no new Manning search.',meaning='Allocated reference flow divided by discharge implied by prescribed width/depth/slope/n. Neither quantity is an independent local flow observation.',thresholds='Factors 10 and 0.1 are descriptive screening, not physical validity limits.',required_depth='Counterfactual consistency calculation, not an adopted geometry correction.',network_note='Do not sum reach Q across serial links or substitute inferred local Q for conserved network allocation.',production_modified=False),indent=2))
    print(out.round(3).to_string(index=False))
    print(f[(f.example=='46') & (f.id_link==67)][['reference_discharge_cms','manning_discharge_cms','allocated_over_manning_q','depth_m','depth_required_for_allocated_q_m']].to_string(index=False))
if __name__=='__main__':main()
