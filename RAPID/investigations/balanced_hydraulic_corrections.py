"""Minimize the largest local roughness correction; diagnostic, no routing edits."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'RAPID/assets/bounded_robust_slopes_20261002'
OUT = BASE / 'common_caps/balanced_fixes'
N0, CMIN, CMAX, X = .035, .05, 5., .1


def evaluate(raw, length, physical_length, margin, clock):
    # Interior bounds dt >= (1+margin)*2KX and
    # dt <= (1-margin)*2K(1-X). Caps impose a finite exhaustive dt range.
    a, b = 2 * X * (1 + margin), 2 * (1 - X) * (1 - margin)
    start = max(1, int(np.ceil(a * length.max() / CMAX)))
    stop = int(np.floor(b * length.min() / CMIN))
    dt = np.arange(start, stop + 1)
    if clock == 'hourly_divisors':
        dt = dt[3600 % dt == 0]
    lower_c = a * length[None, :] / dt[:, None]
    upper_c = b * length[None, :] / dt[:, None]
    # c_used = clip(raw*N0/n, CMIN, CMAX). Invert this piecewise map.
    n_lower = np.where(upper_c >= CMAX, 0., raw * N0 / upper_c)
    n_upper = np.where(lower_c <= CMIN, np.inf, raw * N0 / lower_c)
    target_n = np.clip(N0, n_lower, n_upper)
    log_change = np.abs(np.log(target_n / N0))
    worst = log_change.max(axis=1)
    weighted = log_change @ (physical_length / physical_length.sum())
    idx = np.lexsort((dt, weighted, worst))[0]
    within = {}
    for label, lo, hi in [('ten_pct', .9*N0, 1.1*N0),
                          ('twenty_pct', .8*N0, 1.2*N0),
                          ('n025_050', .025, .05)]:
        within[label] = bool(((n_lower <= hi).all(axis=1)
                             & (n_upper >= lo).all(axis=1)).any())
    return int(dt[idx]), target_n[idx], within, float(np.exp(worst[idx]))


def main():
    OUT.mkdir(exist_ok=True)
    source = BASE / 'common_caps/stability_link_details.csv'
    reference = BASE / 'hydraulic_references.csv'
    data = pd.read_csv(source, dtype={'example': str})
    refs = pd.read_csv(reference, dtype={'example': str}).set_index('example')
    summaries, details = [], []
    for ex, g in data.groupby('example', sort=True):
        ref = refs.loc[ex]
        L = g.effective_length_m.to_numpy()
        physical = g.actual_length_m.to_numpy()
        for alpha in [.65, .625, .60]:
            depth = ref.stored_depth_m * (g.wid_adj_routing.to_numpy()/ref.reference_width_m)**alpha
            raw = 5/(3*N0)*depth**(2/3)*np.sqrt(g.slope.to_numpy())
            original = np.clip(raw, CMIN, CMAX)
            for margin in [0., .01]:
                for clock in ['integer_seconds', 'hourly_divisors']:
                    dt, n, feasible, factor = evaluate(raw, L, physical, margin, clock)
                    c = np.clip(raw*N0/n, CMIN, CMAX)
                    K = L/c
                    assert dt >= ((1+margin)*2*X*K).max()-1e-8
                    assert dt <= ((1-margin)*2*(1-X)*K).min()+1e-8
                    denominator = 2*K*(1-X)+dt
                    coeff = np.array([(dt-2*K*X)/denominator,
                                      (dt+2*K*X)/denominator,
                                      (2*K*(1-X)-dt)/denominator])
                    assert coeff.min() >= -1e-10
                    assert np.allclose(coeff.sum(axis=0), 1.)
                    changed = ~np.isclose(n, N0, rtol=1e-10, atol=0.)
                    row = dict(example=ex, alpha=alpha, margin=margin, clock=clock,
                               dt_s=dt, maximum_n_factor=factor,
                               n_min=n.min(), n_max=n.max(), changed_links=int(changed.sum()),
                               changed_length_pct=100*physical[changed].sum()/physical.sum(),
                               min_c_change_pct=100*(c/original-1).min(),
                               max_c_change_pct=100*(c/original-1).max(),
                               min_coefficient=coeff.min(), **feasible)
                    summaries.append(row)
                    for pos in np.where(changed)[0]:
                        z = g.iloc[pos]
                        details.append(dict(example=ex, alpha=alpha, margin=margin, clock=clock,
                                            dt_s=dt, id_link=int(z.id_link), n_before=N0, n_after=n[pos],
                                            c_before=original[pos], c_after=c[pos],
                                            width_m=z.wid_adj_routing, slope=z.slope,
                                            depth_m=depth[pos], effective_length_m=L[pos],
                                            actual_length_m=physical[pos]))
    result = pd.DataFrame(summaries)
    result.to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(details).to_csv(OUT/'changed_links.csv', index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        date='2026-10-02', script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [source, reference]},
        fixed=dict(X=X, cap_min=CMIN, cap_max=CMAX, nominal_n=N0, segmentation='unchanged'),
        objective='Minimize max(abs(log(n_i/0.035))); tie-break physical-length-weighted sum then dt',
        margin='0 or 1% interior of both dt bounds',
        clock='integer_seconds is nominal; hourly_divisors assumes 3600s forcing, not verified actual clocks',
        scope='16 base networks; no hydrograph runs, child states, or production changes',
        validation='All resulting Muskingum coefficients nonnegative and sum to one; bounds checked'
    ), indent=2))
    print(result[(result.margin == .01)&(result.clock == 'integer_seconds')].to_string(index=False))


if __name__ == '__main__':
    main()
