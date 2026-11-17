"""Full saved-record centroid check for the ten opposite-sign outliers; no routing."""
import json
import numpy as np
import pandas as pd
from investigate_j_outlier_groups import DEST, read_saved, edge_metrics, SOURCES
from incremental_storage_patterns import moment
from incremental_storage import digest


def main():
    cases=pd.read_csv(DEST/'outlier_diagnostics.csv',dtype={'example':str})
    cases=cases[cases.group.str.startswith('G3')]
    rows=[]
    for r in cases.itertuples():
        p=read_saved(r.example,'S000_base','outlet.parquet')
        c=read_saved(r.example,r.state,'outlet.parquet')
        tp=p.time_utc.astype('int64').to_numpy()/1e9;tc=c.time_utc.astype('int64').to_numpy()/1e9
        start=max(tp[0],tc[0]);end=min(tp[-1],tc[-1])
        t=np.unique(np.r_[start,tp[(tp>start)&(tp<end)],tc[(tc>start)&(tc<end)],end])
        qp=np.interp(t,tp,p.q_outlet_cms);qc=np.interp(t,tc,c.q_outlet_cms)
        mp,vp=moment(t,qp-r.baseline_cms);mc,vc=moment(t,qc-r.baseline_cms)
        rawp,_=moment(t,qp);rawc,_=moment(t,qc)
        pp=edge_metrics(read_saved(r.example,'S000_base'));cc=edge_metrics(read_saved(r.example,r.state))
        # Descriptive partition of the exact sum, not a slope intervention.
        low_delta=(cc.loc[cc.slope<1e-5,'mean_contribution_seconds'].sum()-pp.loc[pp.slope<1e-5,'mean_contribution_seconds'].sum())/60
        rows.append(dict(example=r.example,state=r.state,delta_J_min=r.delta_J_minutes,
            window_centroid_min=r.centroid_shift_minutes,full_saved_excess_centroid_min=(mc-mp)/60,
            full_saved_total_flow_centroid_min=(rawc-rawp)/60,start_utc=pd.Timestamp(start,unit='s',tz='UTC'),
            end_utc=pd.Timestamp(end,unit='s',tz='UTC'),record_hours=(end-start)/3600,
            tail_after_window_hours=(end-pd.Timestamp(r.end_utc).timestamp())/3600,
            baseline_cms=r.baseline_cms,parent_initial_cms=qp[0],parent_final_cms=qp[-1],
            child_initial_cms=qc[0],child_final_cms=qc[-1],parent_excess_volume_m3=vp,child_excess_volume_m3=vc,
            slope_below_1e5_delta_J_min=low_delta,remaining_delta_J_min=r.delta_J_minutes-low_delta))
    result=pd.DataFrame(rows);result.to_csv(DEST/'full_saved_record_centroids.csv',index=False)
    (DEST/'full_saved_record_provenance.json').write_text(json.dumps(dict(source_sha256=SOURCES,script_sha256=digest(__file__),
        definition='Exact piecewise-linear centroid, full common saved time span, unchanged original baseline; no clipping, extension, rerouting or extrapolation. Total-flow centroid also supplied separately.',
        limitation='Entire saved record is finite, and final flow need not equal initial flow or baseline. This is not a complete isolated-pulse transfer centroid.',
        slope_partition='Parent and child J sums each partitioned at slope<1e-5. Exact descriptive bookkeeping only; threshold membership can change and this is not a counterfactual intervention.'),indent=2)+'\n')
    print(result[['example','state','delta_J_min','window_centroid_min','full_saved_excess_centroid_min','slope_below_1e5_delta_J_min','remaining_delta_J_min','parent_initial_cms','parent_final_cms']].round(3).to_string(index=False))

if __name__=='__main__':main()
