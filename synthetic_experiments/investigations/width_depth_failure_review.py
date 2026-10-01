"""Summarize alpha feasibility without deleting configurations or changing defaults."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'synthetic_experiments/outputs/width_depth_scaling'

def main():
    g=pd.read_csv(OUT/'grid_preflight.csv');s=pd.read_csv(OUT/'state_preflight.csv');p=pd.read_csv(OUT/'pair_preflight.csv')
    rows=[]
    for (dataset,alpha),a in g.groupby(['dataset','alpha']):
        states=s[(s.dataset==dataset)&(s.alpha==alpha)]
        unique=states.drop_duplicates(['network_id','reference_kb'])
        pairs=p[(p.dataset==dataset)&(p.alpha==alpha)]
        rows.append(dict(dataset=dataset,alpha=alpha,configurations=len(a),failed_configurations=int((~a.feasible).sum()),
            state_instances=len(states),failed_state_instances=int((~states.feasible).sum()),unique_networks=len(unique),failed_unique_networks=int((~unique.feasible).sum()),
            comparisons=len(pairs),failed_pair_clocks=int((~pairs.feasible).sum()),
            failed_geometric_combinations=a[~a.feasible][['outer_a_fraction','length_multiplier']].drop_duplicates().shape[0]))
    pd.DataFrame(rows).to_csv(OUT/'failure_summary.csv',index=False)
    bad=g[~g.feasible].groupby(['dataset','alpha','outer_a_fraction','length_multiplier']).agg(failed_configurations=('configuration_id','size'),extents_m=('embedded_extent_m',lambda v:','.join(str(int(x)) for x in sorted(v.unique())))).reset_index()
    bad.to_csv(OUT/'failed_parameter_combinations.csv',index=False)
    s[~s.feasible].groupby(['dataset','alpha','state']).size().rename('failed_state_instances').reset_index().to_csv(OUT/'failed_states_by_type.csv',index=False)
    policies=[]
    for alpha,a in g[g.dataset=='embedded_width_length_extension'].groupby('alpha'):
        masks={'keep_all':pd.Series(True,index=a.index),
            'exclude_A005_m3':~((a.outer_a_fraction==.05)&(a.length_multiplier==3)),
            'exclude_A005_m2_or_3':~((a.outer_a_fraction==.05)&(a.length_multiplier>=2)),
            'require_A_at_least_010':a.outer_a_fraction>=.1}
        for policy,mask in masks.items():
            ids=a.loc[mask,'configuration_id'];pairs=p[(p.dataset=='embedded_width_length_extension')&(p.alpha==alpha)&p.configuration_id.isin(ids)]
            policies.append(dict(alpha=alpha,policy=policy,retained_configurations=int(mask.sum()),removed_configurations=int((~mask).sum()),
                remaining_failed_configurations=int((~a.loc[mask,'feasible']).sum()),retained_comparisons=len(pairs),remaining_failed_pair_clocks=int((~pairs.feasible).sum())))
    pd.DataFrame(policies).to_csv(OUT/'exclusion_scenarios.csv',index=False)
    real=pd.read_csv(OUT/'real_base_preflight.csv',dtype={'example':str});edges=pd.read_csv(OUT/'real_edge_preflight.csv',dtype={'example':str});controllers=[]
    for _,r in real.iterrows():
        slope_path=ROOT/f"network_variants/output/{r.example}_indep__anchored/states/S000_base/hydraulic_context/link_hydraulic_context.csv"
        slopes=pd.read_csv(slope_path).set_index('id_link')
        for role in ['slow','fast']:
            link=int(r[role+'_link_id']);e=edges[(edges.example==r.example)&(edges.alpha==r.alpha)&(edges.id_link==link)].iloc[0].to_dict()
            source=slopes.loc[link]
            controllers.append(dict(e,role=role,network_feasible=r.feasible,width_refresh_marker=r.width_refresh_marker,
                dt_lower_seconds=r.dt_lower_seconds,dt_upper_seconds=r.dt_upper_seconds,
                slope_source_method=source.get('slope_source_method'),slope_reason=source.get('slope_reason'),slope_section_ref=source.get('slope_section_ref'),
                raw_slope=source.get('raw_slope'),bubble_reference_slope=source.get('bubble_reference_slope'),bubble_slope_repair_applied=source.get('bubble_slope_repair_applied')))
    pd.DataFrame(controllers).to_csv(OUT/'real_controllers.csv',index=False)
    domain=edges[edges.alpha==.6].groupby('example').agg(n_links=('id_link','size'),min_width_fraction=('width_over_reference','min'),p05_width_fraction=('width_over_reference',lambda v:v.quantile(.05)),min_slope_ratio=('slope_over_reference','min'),max_slope_ratio=('slope_over_reference','max'),blocking_links=('blocks_fast_bound','sum'))
    domain.to_csv(OUT/'real_geometry_domain.csv')
    sources=['grid_preflight.csv','state_preflight.csv','pair_preflight.csv','real_base_preflight.csv','real_edge_preflight.csv','real_sources.json']
    provenance=dict(reference='HYD-2026-09-29-ALPHA06-FAILURE-REVIEW',inputs={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in sources},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unique_network_definition='Canonical directed network geometry plus reference kb within dataset; repeated treatment/configuration references deduplicated, no clock included.',
        failure_definition='No common integer timestep satisfying nonnegative Muskingum coefficients; distinguish full configuration, individual state and parent-child pair clocks.',
        exclusions='Counterfactual summaries only; no configurations removed. Width fraction is a branch/reference-section ratio, not a bifurcation flow split in real data.')
    (OUT/'failure_review_manifest.json').write_text(json.dumps(provenance,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(bad.to_string(index=False))
if __name__=='__main__':main()
