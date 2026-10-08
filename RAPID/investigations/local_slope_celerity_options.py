"""Audit slope equivalents and local observation context; no hydraulic edits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
OUT = BASE/'common_caps/local_slope_celerity'


def main():
    OUT.mkdir(exist_ok=True)
    links = pd.read_csv(BASE/'common_caps/fix_options/local_equivalent_changes.csv', dtype={'example':str})
    links['candidate_slope'] = links.slope * links.slope_factor_equivalent
    links['current_drop_m'] = links.slope * links.actual_length_m
    links['candidate_drop_m'] = links.candidate_slope * links.actual_length_m
    links['extra_drop_m'] = links.candidate_drop_m-links.current_drop_m
    observations = []
    for ex, group in links.groupby('example'):
        folder = BASE/ex/'floor_0.05'
        obs = pd.read_parquet(folder/'observations.parquet')
        profile = pd.read_parquet(folder/'profile.parquet').sort_values('reference_section_dist_out_m')
        px, py = profile.reference_section_dist_out_m.to_numpy(), profile.fitted_wse_m.to_numpy()
        for r in group.itertuples():
            left, right = sorted([r.node_us_reference_km*1000, r.node_ds_reference_km*1000])
            for window, buffer in [('exact',0),('expanded_1km',1000)]:
                lo, hi = max(px.min(),left-buffer), min(px.max(),right+buffer)
                if hi <= lo:
                    continue
                selected = obs[obs.reference_section_dist_out_m.between(lo,hi)]
                for aid, g in selected.groupby('acquisition_id'):
                    q = g.groupby('reference_section_dist_out_m',as_index=False).wse_m.mean()
                    x, y = q.reference_section_dist_out_m.to_numpy(), q.wse_m.to_numpy()
                    if len(x)<4 or np.ptp(x)<.5*(hi-lo):
                        continue
                    observed = theilslopes(y,x).slope
                    fitted = theilslopes(np.interp(x,px,py),x).slope
                    observations.append(dict(example=ex,id_link=r.id_link,window=window,
                                             acquisition_id=aid,cycle=g.cycle.iloc[0],
                                             observed_slope=observed,matched_profile_slope=fitted,
                                             delta_slope=observed-fitted,n_positions=len(x)))
    o = pd.DataFrame(observations)
    o.to_csv(OUT/'acquisition_context.csv',index=False)
    cycles = o.groupby(['example','id_link','window','cycle'],as_index=False)[
        ['observed_slope','matched_profile_slope','delta_slope']].median()
    rows=[]
    for (ex, link, window), g in cycles.groupby(['example','id_link','window']):
        rows.append(dict(example=ex,id_link=link,window=window,n_cycles=len(g),
                         observed_median=g.observed_slope.median(),
                         observed_q25=g.observed_slope.quantile(.25),
                         observed_q75=g.observed_slope.quantile(.75),
                         matched_profile_median=g.matched_profile_slope.median(),
                         fraction_steeper=(g.delta_slope>1e-10).mean()))
    context=pd.DataFrame(rows)
    context.to_csv(OUT/'cycle_context.csv',index=False)
    links.to_csv(OUT/'link_options.csv',index=False)
    columns=['example','id_link','ownership_kind','slope','candidate_slope',
             'slope_factor_equivalent','extra_drop_m','celerity_mps','target_celerity_mps']
    worst=links.loc[links.groupby('example').slope_factor_equivalent.idxmax(),columns]
    print(worst.to_string(index=False))
    print(context.merge(worst[['example','id_link']],on=['example','id_link']).to_string(index=False))


if __name__=='__main__':
    main()
