"""Readable report and scientific figures for the signed-coefficient pilot."""
from pathlib import Path
import json,base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/common_caps/signed_coefficient_check'


def main():
    metrics=pd.read_csv(OUT/'comparisons.csv',dtype={'example':str})
    runs=pd.read_csv(OUT/'routing_summary.csv',dtype={'example':str})
    qa=json.loads((OUT/'segment_validation.json').read_text())
    assert len(qa)==4,'Wait for both fine-resolution segment audits'
    stats=[]
    for ex,dt,tag in [('11',69,''),('11',23,'_dt23'),('46',35,''),('46',7,'_dt7')]:
        d=pd.read_csv(OUT/f'{ex}{tag}_every_segment_audit.csv')
        for name,g in d.groupby('forcing'):
            stats.append(dict(example=ex,dt_s=dt,forcing=name,n_segments=len(g),
                min_discharge=g.minimum_cms.min(),
                normalized_min_pct=100*g.minimum_divided_by_branch_fraction.min() if name!='actual' else np.nan,
                normalized_max_pct=100*(g.maximum_cms/g.branch_fraction).max() if name!='actual' else np.nan,
                branch_normalized_min_actual_cms=g.minimum_divided_by_branch_fraction.min() if name=='actual' else np.nan,
                max_negative_volume_pct=100*g.negative_volume_fraction.max(),
                segments_with_q_below_minus_1e_minus8=int((g.minimum_cms< -1e-8).sum())))
    stats=pd.DataFrame(stats);stats.to_csv(OUT/'segment_audit_summary.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for ax,(ex,state) in zip(axes,[('11','S003_unit_18'),('46','S004_unit_24'),('46','S010_unit_105')]):
        g=metrics[metrics.example.eq(ex)&metrics.state.eq(state)].sort_values('dt_s')
        ax.plot(g.dt_s,g.timing_change_minutes,'o-',color='#2563eb')
        center=g.timing_change_minutes.mean();span=max(.1,np.ptp(g.timing_change_minutes)*2)
        ax.set_ylim(center-span,center+span)
        ax.set(title=f'Case {ex}, unit {state.split("_")[-1]}',xlabel='Routing timestep (s)',ylabel='Child − parent peak time (min)')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False);ax.grid(alpha=.2)
    fig.suptitle('Fixed hydraulic parameters: paired timing converges')
    fig.savefig(OUT/'paired_timing_convergence.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for tag,dt,color in [('',35,'#2563eb'),('_dt7',7,'#ea580c')]:
        h=pd.read_parquet(OUT/f'46{tag}_worst_rectangle_segment.parquet')
        center=h.time_seconds.iloc[h.local_output.argmin()]
        mask=h.time_seconds.between(center-1800,center+3600)
        axes[0].plot((h.time_seconds[mask]-86400)/3600,h.local_input[mask],ls=':',color=color,alpha=.7,label='Local input (dotted)' if dt==35 else None)
        axes[0].plot((h.time_seconds[mask]-86400)/3600,h.local_output[mask],color=color,label=f'Output, dt={dt} s')
        axes[1].plot((h.time_seconds[mask]-86400)/3600,100*h.local_output[mask],color=color,label=f'dt={dt} s')
    axes[0].set(title='Case 46: worst segment, sharp pulse',ylabel='Flow / branch-scaled pulse amplitude')
    axes[1].set(title='Zoom: artificial initial dip',ylabel='Percent of branch-scaled pulse amplitude',ylim=(-.8,.2))
    for ax in axes:ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Hours after upstream pulse begins');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'negative_dip_explanation.png',dpi=170);plt.close(fig)
    rows=['| Case | dt (s) | Actual: minimum discharge anywhere | Smooth pulse: worst dip | Sharp pulse: worst dip |',
          '|---|---:|---:|---:|---:|']
    for (ex,dt),g in stats.groupby(['example','dt_s']):
        g=g.set_index('forcing')
        rows.append(f'| {ex} | {dt} | {g.loc["actual","min_discharge"]:.3f} m³/s | {g.loc["triangle_12h_24h","normalized_min_pct"]:.6g}% | {g.loc["rectangle_24h","normalized_min_pct"]:.6g}% |')
    comparison_rows=['| Case / collapse | Timesteps (s) | Peak timing change (min) | Duration change (%) | Peak-excess change (%) |',
                     '|---|---|---|---|---|']
    for (ex,state),g in metrics.groupby(['example','state']):
        comparison_rows.append(f'| {ex} / {state} | '+', '.join(str(int(x)) for x in g.dt_s)+' | '+', '.join(f'{x:.5f}' for x in g.timing_change_minutes)+f' | {g.duration_change_pct.min():.6f} to {g.duration_change_pct.max():.6f} | {g.peak_excess_change_pct.min():.6f} to {g.peak_excess_change_pct.max():.6f} |')
    maxmass=max(abs(x) for r in qa for x in r['mass_residual_relative'])
    report='''# Signed-coefficient routing pilot — 2026-10-02

Reference: SIGNED-MUSKINGUM-2026-10-02. Requested after explicit timestep-based
celerity correction made results depend on the selected timestep. This test
keeps hydraulic K values fixed and separates boundedness from coefficient
positivity. It is an investigation, not a production-method promotion.

## What was held fixed

Original bounded robust slopes; alpha 0.65; n 0.035; X 0.1; 200 m segmentation
and effective length floor; common 0.05–5 m/s celerity caps. No local 5 cm head
adjustments and no additional timestep-dependent celerity corrections.
The actual forcing, child geometry, shared-head mapping and plateau event
definition are those documented in the preceding explicit-regularization pilot.

Case 11 is tested at 69, 46 and 23 seconds; case 46 at 35, 31 and 7 seconds.
Each clock divides the saved forcing interval and is shared by the selected
parent and children. All clocks satisfy dt <= 2K(1-X) for every segment, but
may violate dt >= 2KX. Fifteen actual-forcing hydrographs were computed.
At both largest and smallest clocks, EVERY base-network subreach was also
routed directly in time for actual forcing and two controlled zero-base
pulses: a 12-hour rise/24-hour fall triangle, and a 24-hour rectangular pulse.
There are 419 segments in case 11 and 6,822 in case 46. Internal child segments
were not separately audited; their outlet responses were tested at all clocks.

## Meaning of the negative allowance

The recurrence is O_new = a I_new + b I_old + c O_old, with
a=(dt-2KX)/(dt+2K(1-X)). Allowing a<0 retains the hydraulic K value but permits
an inverse initial response: a rising inflow can initially lower the outflow.
For link 11/52 at dt=69 s, a=-0.0820, b=0.1344 and c=0.9477. If local inflow
instantaneously rises from 1000 to 1100 m³/s from equilibrium, the ordinary
internal-reach recurrence gives about 991.8 m³/s initially: an artificial
8.2 m³/s dip. Actual incoming hydrographs at that link are smoother.
The external inlet segment uses RAPID's held-forcing variant of the equation.

Keeping 0<=c<1 prevents an alternating homogeneous decay and ensures that
the single-segment free response decays. It does not make the response
positivity-preserving. Shorter timesteps can make a more negative, so temporal
refinement need not remove an initial undershoot. Stability, positivity and
physical accuracy must not be used interchangeably.

This also follows from the continuous fixed-parameter Muskingum approximation:
K(1-X) dO/dt + O = I - KX dI/dt. A sufficiently abrupt positive inflow
derivative can drive the right-hand side downward. Thus convergence in dt
does not guarantee removal of this inverse-response feature of the chosen
model at fixed segmentation/K/X. Slow-varying forcing is much less demanding.

## Results: comparative timing and shape

'''+ '\n'.join(comparison_rows)+'''

Peak timing uses the three-point quadratic estimate to avoid timestep-bin
jumps. The numerical agreement does not imply subsecond physical accuracy.
Case 11/unit 18 varies by about 0.13 seconds across the tested clocks; both
case 46 timing contrasts vary by less than 0.01 seconds. This contrasts with
the earlier clock-dependent celerity correction, especially for unit 24.
Base peak lag ranges from 27.838 to 27.882 hours in case 11 and 339.449 to
339.456 hours in case 46. Absolute lags vary more than paired contrasts.
All three tested duration/peak-excess contrasts are also effectively unchanged
at the displayed precision. This is a selected-case result, not all collapses.

## Every-segment behavior

Controlled-pulse dips are normalized by the upstream unit pulse amplitude
times the segment's steady bifurcation allocation fraction. Actual discharge
minima are absolute m³/s, across all segments. Zero-base stress inputs expose
negative perturbations which a large river baseflow could otherwise conceal.

'''+ '\n'.join(rows)+f'''

No negative discharge was found in either actual-forcing base network at the
two fully audited clocks. The controlled pulses do produce small signed
transients. In case 46 the sharp pulse has a local dip of roughly half a
percent, whereas the smooth pulse has a dip of order 0.0015%. These are
localized artifacts, not growing instability; they persist under refinement.
The worst sharp-pulse segment is identified in `segment_validation.json`.
Small post-pulse oscillations/overshoots are also recorded in the segment
tables; do not characterize signed routing as strictly oscillation-free.

Integrated inflow minus trapezoidal outlet flow matches the change in the
storage represented by the implemented Muskingum recurrence. Maximum absolute
relative mass residual across the actual/pulse audits is {maxmass:.3g}.
No negative values were clipped. Actual-input solutions remain inside the
global input range after normalization by steady branch allocation in the
coarse-clock audits; this does not prove local monotonicity at every instant.

## Important scope qualification

This is NOT permission to violate the rule on only the earlier 77 links.
At dt=35 s, 452 of 541 original links in case 46 have a negative current-inflow
coefficient; at dt=7 s, all 541 do. Many still produce entirely well-behaved
smooth-flow responses. The relevant acceptance evidence is the response and
convergence audit, not simply a small count of negative coefficients.

The evidence supports continuing with fixed-hydraulic signed-coefficient
routing as a candidate for these slow-varying events. It does not establish
positivity for abrupt/zero-flow conditions, spatial convergence, field accuracy
or performance for every real-network collapse. Production defaults and
coefficient-positivity guards remain unchanged. Before adoption, repeat the
audit for the intended full comparison set and record the event bandwidth
and observed artifact amplitudes rather than choosing an undocumented tolerance.

## Reproduction and references

- Runner: `RAPID/investigations/signed_coefficient_routing_check.py`.
- Reporter: `RAPID/investigations/report_signed_coefficient_check.py`.
- `routing_summary.csv`, `comparisons.csv`, `coefficients.csv`.
- `*_every_segment_audit.csv`: every base segment at coarse/fine clocks.
- `segment_audit_summary.csv`, `segment_validation.json`: signed dips,
  negative volume, mass residuals and independent recurrence verification.
- Outlet and worst-segment series are saved as Parquet; figures are PNG.
- The frequency-domain calculation is checked against the time-domain RAPID
  recurrence with an explicitly negative-coefficient impulse, and against
  all-segment time-domain runs on both real base networks.
- [HEC Muskingum equations](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/channel-flow/muskingum-model).
- [Ponce and Theurer: Accuracy Criteria in Diffusion Routing](https://ponce.sdsu.edu/accuracy_criteria_in_diffusion_routing.html),
  distinguishes numerical stability from accuracy and negative-flow artifacts.
'''
    (OUT/'README.md').write_text(report)
    notebook=ROOT/'RAPID/notebooks/slope_smoothing_floor_comparison.ipynb'
    nb=nbformat.read(notebook,as_version=4)
    marker='SIGNED-MUSKINGUM-2026-10-02'
    # Replace this report and its attachment cells if rerun.
    nb.cells=[c for c in nb.cells if marker not in c.source]
    nb.cells.append(nbformat.v4.new_markdown_cell(report.replace('# Signed-coefficient','## Signed-coefficient',1)))
    for name in ['paired_timing_convergence.png','negative_dip_explanation.png']:
        cell=nbformat.v4.new_markdown_cell(f'<!-- {marker} -->\n![{name}](attachment:{name})')
        cell.attachments={name:{'image/png':base64.b64encode((OUT/name).read_bytes()).decode()}}
        nb.cells.append(cell)
    nbformat.validate(nb);nbformat.write(nb,notebook)
    print(stats.to_string(index=False));print('Saved report and notebook')


if __name__=='__main__':main()
