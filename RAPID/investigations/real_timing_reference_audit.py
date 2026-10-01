"""Read-only gauge/reference audit. Reference scenarios are NOT routed forecasts."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'RAPID/src'))
from rapid_tools.analysis import _load_gauge_quality_sheet, _load_station_timeseries_for_observed_lag
from rapid_tools.speed_reference import build_speed_reference_frame

OUT = ROOT / 'RAPID/assets/real_timing_reference_audit_20260930'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / 'network_variants/output').glob('*_indep__anchored/rapid_prep_manifest.json'))
    ids = [p.parent.name.split('_')[0] for p in paths]
    refs = build_speed_reference_frame(ids, output_root=ROOT / 'network_variants/output').set_index('example_id')
    quality = _load_gauge_quality_sheet()
    rows, scenarios, sources = [], [], {}
    for ex, path in zip(ids, paths):
        m = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        cfg, ref = m['prep_config'], m['reference_kb_summary']
        row = dict(example_id=ex, **refs.loc[ex].to_dict())
        qr = quality.loc[pd.to_numeric(quality.Example, errors='coerce').eq(int(ex))].iloc[0]
        row.update(peak_quality=qr.Peak_quality, peak_match=qr.Peak_match, quality_reason=qr.Reason,
                   stored_n=cfg['n_manning'], width_refresh_marker=(path.parent/'rapid_width_refresh_required.json').exists())
        start, end = pd.to_datetime(qr.Period_start, utc=True), pd.to_datetime(qr.Period_end, utc=True)
        for side, station in [('up', qr.Up), ('down', qr.Down)]:
            data = _load_station_timeseries_for_observed_lag(str(station))
            data = data[data.time.between(start, end)].copy()
            if data.empty:
                continue
            peak = data.loc[data.discharge_cms.idxmax()]
            row[side+'_peak_utc'] = peak.time.isoformat()
            row[side+'_sampling_median_hours'] = data.time.diff().dt.total_seconds().median()/3600
            row[side+'_peak_at_window_edge'] = bool(peak.name in [data.index[0], data.index[-1]])
            row[side+'_equal_max_count'] = int(data.discharge_cms.eq(peak.discharge_cms).sum())
            equal = data.loc[data.discharge_cms.eq(peak.discharge_cms), 'time']
            row[side+'_equal_max_first_utc'] = equal.min().isoformat()
            row[side+'_equal_max_last_utc'] = equal.max().isoformat()
            row[side+'_equal_max_span_hours'] = (equal.max()-equal.min()).total_seconds()/3600
        row['equal_max_lag_min_hours'] = (pd.to_datetime(row['down_equal_max_first_utc'])-pd.to_datetime(row['up_equal_max_last_utc'])).total_seconds()/3600
        row['equal_max_lag_max_hours'] = (pd.to_datetime(row['down_equal_max_last_utc'])-pd.to_datetime(row['up_equal_max_first_utc'])).total_seconds()/3600
        hp = path.parent/'states/S000_base/rapid/run/outlet_hydrograph.csv'
        if hp.exists():
            h = pd.read_csv(hp)
            ti = h.loc[h.q_inflow_cms.idxmax()]
            to = h.loc[h.q_outlet_cms.idxmax()]
            row['stored_routed_lag_hours'] = (to.time_seconds-ti.time_seconds)/3600
            row['forcing_peak_offset_from_gauge_hours'] = (pd.to_datetime(ti.time_utc)-pd.to_datetime(row['up_peak_utc'])).total_seconds()/3600
            row['stored_lag_ratio_to_observed'] = row['stored_routed_lag_hours']/row['observed_lag_hours']
            sources[str(hp.relative_to(ROOT))] = hashlib.sha256(hp.read_bytes()).hexdigest()
        w, depth, slope = [float(ref[k]) for k in ['reference_width_m', 'reference_depth_m', 'reference_slope']]
        numerator = (5/3)*depth**(2/3)*np.sqrt(slope)
        row.update(reference_width_m=w, reference_depth_m=depth, reference_slope=slope,
                   n_matching_gauge_speed_at_reference=numerator/row['reference_speed_mps'])
        mapping = path.parent/'states/S000_base/rapid/run/rapid_link_mapping.parquet'
        d = pd.read_parquet(mapping) if mapping.exists() else None
        for n in [.35, .025, .035, .05]:
            c = numerator/n
            used = np.clip(c, cfg['min_celerity_mps'], cfg['max_celerity_mps'])
            s = dict(example_id=ex, n=n, reference_celerity_raw_mps=c, reference_celerity_capped_mps=used,
                     observed_speed_mps=row['reference_speed_mps'], observed_lag_hours=row['observed_lag_hours'],
                     reference_uniform_time_hours=row['reference_length_m']/used/3600,
                     uncapped_reference_uniform_time_hours=row['reference_length_m']/c/3600,
                     raw_speed_error_pct=100*(c/row['reference_speed_mps']-1),
                     capped_speed_error_pct=100*(used/row['reference_speed_mps']-1))
            s['uniform_time_error_pct'] = 100*(s['reference_uniform_time_hours']/row['observed_lag_hours']-1)
            s['uncapped_uniform_time_error_pct'] = 100*(s['uncapped_reference_uniform_time_hours']/row['observed_lag_hours']-1)
            if d is not None:
                # Frozen prepared geometry; separate alpha sensitivity, not adopted real-case settings.
                for alpha in [1., .65]:
                    local_depth = depth*(d.rapid_width_m/w)**alpha
                    raw = (5/3)*local_depth**(2/3)*np.sqrt(d.slope_used)/n
                    weight = d.link_length_m
                    tag = 'alpha1' if alpha == 1 else 'alpha065'
                    s[tag+'_length_fraction_lower_capped'] = np.average(raw < cfg['min_celerity_mps'], weights=weight)
                    s[tag+'_length_fraction_upper_capped'] = np.average(raw > cfg['max_celerity_mps'], weights=weight)
            scenarios.append(s)
        rows.append(row)
        print('Audited', ex, flush=True)
    cases = pd.DataFrame(rows)
    sensitivity = pd.DataFrame(scenarios)
    cases.to_csv(OUT/'cases.csv', index=False)
    sensitivity.to_csv(OUT/'reference_scenarios.csv', index=False)
    summary = []
    for n, g in sensitivity.groupby('n'):
        valid = g[np.isfinite(g.uniform_time_error_pct) & (g.observed_lag_hours > 0)]
        summary.append(dict(n=n, count=len(valid), median_abs_time_error_pct=valid.uniform_time_error_pct.abs().median(),
                            within_25pct=int(valid.uniform_time_error_pct.abs().le(25).sum()),
                            within_50pct=int(valid.uniform_time_error_pct.abs().le(50).sum()),
                            uncapped_within_25pct=int(valid.uncapped_uniform_time_error_pct.abs().le(25).sum()),
                            uncapped_median_abs_time_error_pct=valid.uncapped_uniform_time_error_pct.abs().median(),
                            median_signed_time_error_pct=valid.uniform_time_error_pct.median()))
    pd.DataFrame(summary).to_csv(OUT/'reference_summary.csv', index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='REAL-TIMING-2026-09-30',
        scope='All available independent anchored preparations; no production settings changed and no new routing.',
        warning='Uniform reference times are screening estimates, not network peak forecasts. Gauge speeds are apparent event propagation speeds, not local hydraulic measurements.',
        old_run_note='Saved hydrographs may predate width/slope refresh. Current prep config is recorded, not proof of run/prep identity.',
        observed_lag_method='Difference of first discharge maxima in the quality-sheet event window. Equal-maximum lag envelopes are ambiguity bounds, not confidence intervals. Saved sampling can include interpolation.',
        branch_sensitivity='Frozen mapping widths/slopes with reference-anchored depths; alpha=.65 only a scenario, not an adopted real-case choice.',
        source_sha256=sources, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for n, g in sensitivity.groupby('n'):
        axes[0].scatter(g.observed_lag_hours, g.reference_uniform_time_hours, label=f'n={n:g}')
    axes[1].scatter(cases.observed_lag_hours, cases.stored_routed_lag_hours)
    for _, r in cases.iterrows():
        axes[1].annotate(r.example_id, (r.observed_lag_hours, r.stored_routed_lag_hours), fontsize=8)
    for ax in axes:
        ax.plot([1, 10000], [1, 10000], color='gray', linestyle='--', linewidth=1)
        ax.set(xscale='log', yscale='log', xlabel='Observed gauge peak lag (h)')
        ax.grid(alpha=.15)
    axes[0].set(title='Uniform reference estimates — not routed results', ylabel='Reference distance / capped celerity (h)')
    axes[0].legend()
    axes[1].set(title='Saved historical network runs', ylabel='Model inlet-to-outlet peak lag (h)')
    fig.savefig(OUT/'timing_audit.png', dpi=180)
    plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
