"""Summarize the completed parent-only Cunge viability experiment."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/cunge_calibration_alpha2'


def main():
    results=pd.read_csv(OUT/'comparison.csv',dtype={'example':str})
    assert len(results)==32 and results.example.nunique()==16
    paths={r.example:Path(r.output_dir) for r in results.itertuples()}
    full=pd.DataFrame([json.loads((p/'full_engine_verification.json').read_text()) for p in paths.values()])
    full.to_csv(OUT/'prepared_verification.csv',index=False)
    pulses=pd.concat([pd.read_csv(p/'fitted_pulse_checks.csv',dtype={'example':str}) for p in paths.values()])
    pulses.to_csv(OUT/'fitted_pulse_checks.csv',index=False)
    independent=[]
    for p in (OUT/'independent_events').glob('*_results.csv'):
        try:independent.append(pd.read_csv(p,dtype={'example':str}))
        except pd.errors.EmptyDataError:pass
    hold=pd.concat(independent,ignore_index=True);hold.to_csv(OUT/'independent_verification.csv',index=False)
    m=results[results.method.eq('cunge')].set_index('example');b=results[results.method.eq('fixed_x')].set_index('example')
    summary=results.groupby('method').agg(fitted=('within_sampling','sum'),n_min=('n_manning','min'),n_max=('n_manning','max'),
             n_median=('n_manning','median'),band_fitted=('common_band_within_sampling','sum'),
             median_shape_rmse=('normalized_shape_rmse','median'),median_abs_timing_error_h=('abs_error_h','median'),max_abs_timing_error_h=('abs_error_h','max'))
    summary.to_csv(OUT/'calibration_summary.csv')
    usable=hold[hold.valid]
    paired=hold[hold.method.eq('cunge')].merge(hold[hold.method.eq('fixed_x')],on=['example','event'],suffixes=('_cunge','_fixed'))
    paired=paired[paired.valid_cunge&paired.valid_fixed].copy()
    paired['timing_improvement_h']=paired.abs_error_h_fixed-paired.abs_error_h_cunge
    paired['shape_improvement']=paired.normalized_shape_rmse_fixed-paired.normalized_shape_rmse_cunge
    paired.to_csv(OUT/'independent_comparison.csv',index=False)
    closure=[];audit=[];wave_checks=[]
    for ex,p in paths.items():
        f=pd.read_parquet(p/'cunge_prep/hydraulic_links.parquet');case=json.loads((p/'case.json').read_text())
        q_normal=f.width*f.depth_m*(.6*f.rapid_celerity_mps_raw)
        ratio=f.reference_discharge_cms/q_normal
        closure.append(dict(example=ex,reference_to_wide_channel_manning_Q_median=float(ratio.median()),
                            reference_to_wide_channel_manning_Q_min=float(ratio.min()),reference_to_wide_channel_manning_Q_max=float(ratio.max()),
                            caveat='Wide rectangular, prescribed-depth diagnostic; uncapped Manning c; not observed discharge error.'))
        k=f.rapid_k.to_numpy();x=f.rapid_x.to_numpy();dt=case['routing_dt_s'];den=dt+2*k*(1-x);a=(dt-2*k*x)/den;bb=(dt+2*k*x)/den;d=(2*k*(1-x)-dt)/den
        gain=((a+bb)/(1-d))**f['count'].to_numpy()
        audit.append(dict(example=ex,original_links=len(f),sections=int(f['count'].sum()),
                          physical_length_m=float(f.link_length_m.sum()),represented_length_m=float((f['count']*f.rapid_effective_length_m).sum()),
                          max_cell_to_Ld=float((f.rapid_effective_length_m/f.hydraulic_length_scale_m).max()),
                          max_edge_dc_gain_error=float(abs(gain-1).max()),min_previous_outflow_coefficient=float(d.min())))
        base=pd.read_parquet(p/'cunge_best_wave.parquet')
        fine=pd.read_parquet(p/f'cunge_n{m.loc[ex].n_manning:.12f}_dt1_grid2.parquet')
        z=base.merge(fine,on='time_seconds',suffixes=('_base','_fine'))
        wave_checks.append(dict(example=ex,max_grid_doubling_wave_difference_fraction=float(
            abs(z.q_outlet_cms_base-z.q_outlet_cms_fine).max()/np.ptp(base.q_inflow_cms)),
            compared_samples=len(z),base_samples=len(base),fine_samples=len(fine)))
    pd.DataFrame(closure).to_csv(OUT/'hydraulic_consistency_diagnostic.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'grid_and_gain_audit.csv',index=False)
    pd.DataFrame(wave_checks).to_csv(OUT/'wave_convergence.csv',index=False)
    figure_calibration(results,paths)
    figure_holdouts(paired)
    figure_hydrographs(results,paths)
    matched=m[m.within_sampling].index.intersection(b[b.within_sampling].index)
    conclusion=dict(calibration_summary=summary.reset_index().to_dict('records'),
                    matched_successful_networks=len(matched),
                    matched_fixed_n_range=[float(b.loc[matched].n_manning.min()),float(b.loc[matched].n_manning.max())],
                    matched_cunge_n_range=[float(m.loc[matched].n_manning.min()),float(m.loc[matched].n_manning.max())],
                    independent_selected_events=int(hold.event.nunique()),independent_usable_pairs=len(paired),
                    independent_timing_improved=int((paired.timing_improvement_h>1e-9).sum()),
                    independent_timing_worse=int((paired.timing_improvement_h< -1e-9).sum()),
                    independent_shape_improved=int((paired.shape_improvement>1e-9).sum()),
                    independent_shape_worse=int((paired.shape_improvement< -1e-9).sum()),
                    independent_median_abs_timing_error_h={'fixed_x':float(paired.abs_error_h_fixed.median()),'cunge':float(paired.abs_error_h_cunge.median())},
                    independent_median_normalized_shape_rmse={'fixed_x':float(paired.normalized_shape_rmse_fixed.median()),'cunge':float(paired.normalized_shape_rmse_cunge.median())},
                    max_full_sparse_cascade_difference_cms=float(full.max_engine_cascade_error_cms.max()),
                    max_time_halving_peak_change_s=float(results.half_clock_lag_change_s.abs().max()),
                    max_cunge_grid_doubling_peak_change_s=float(m.fine_grid_lag_change_s.abs().max()),
                    max_cunge_grid_doubling_peak_discharge_change_pct=float(m.fine_grid_peak_change_pct.abs().max()),
                    max_cunge_grid_doubling_wave_difference_percent=100*max(r['max_grid_doubling_wave_difference_fraction'] for r in wave_checks),
                    max_fitted_pulse_dip_percent=float(100*pulses.maximum_pulse_dip_fraction.max()),
                    min_pulse_finite_window_volume_ratio=float(pulses.finite_window_excess_volume_ratio.min()),
                    min_actual_native_flow_cms=float(full.minimum_substep_flow_cms.min()),
                    positive_finite_discharge_all_full_runs=bool((full.minimum_substep_flow_cms>=0).all()),
                    all_new_inputs_separate=True,children_rerun=False)
    (OUT/'summary.json').write_text(json.dumps(conclusion,indent=2))
    text='''CONSTANT-REFERENCE MUSKINGUM–CUNGE, ALPHA=2: PARENT VIABILITY ASSESSMENT

SCOPE
Implements and evaluates agreed steps1–4. All16 base networks were recalibrated; no child-network calibration or rerouting was performed. Existing state directories, historical results and production defaults remain in place. The new engine path is explicit and opt-in.

IMPLEMENTATION
Actual anchored link lengths; no artificial length floor. Frozen reference discharge allocated by width-based conservative splits and summed at joins. Prescribed existing depth-width relation, slopes and widths; c=(5/(3n))*h^(2/3)*sqrt(S), capped at0.05–5 m/s as before. Q, geometry and depth do not change during calibration.
Ld=Q/(B*S*c); N=max(1,ceil(L/(2*Ld_fastest))) using the fastest c across n=0.010–0.100. Equal numerical sections retain exact physical length. Thus dx<=2Ld across the full n range, X<=0.25, but negative X is permitted. Every n trial recomputes c, K=dx/c and X=0.5*(1-Ld/dx).
Only Manning n is fitted. Alpha, reference discharge, geometry and celerity caps are prescribed, not optimized. Grid and clock are fixed across both methods and all calibration trials. The common clock retains dt<=2K(1-X) but does not impose nonnegative inflow coefficients. Native negative flow and undershoot are audited without clipping.
The explicit upstream boundary uses current and previous upstream inflow; subsequent native substeps use the same held forcing. Steady first-flow initialization remains. No lateral input. New connectivity uses conservative float64 ratios; legacy engine defaults preserve their earlier precision and boundary mode.
Numerical prep bundles contain rat_srt.csv, riv.csv, kfc.csv, xfc.csv, inflow.nc, forcing_normalized.csv, hydraulic_links.parquet, rapid_link_attributes.csv and rapid_prep_manifest.json. Mapping records original physical links, numerical subsections and anchor nodes. GIS geometry is referenced from the source anchored state; it is not regenerated by this prep entry point.

CALIBRATION DESIGN
The current reviewed workbook windows and first-sampled-maximum observed travel-time targets were reused. The objective remains absolute peak-lag error; hydrograph amplitude/shape were not fitted. The range0.010–0.100 is diagnostic, not a claim of site-plausible roughness. A0.025–0.060 comparison band is reported separately. Coarse grid, best-neighborhood refinement and observed sign crossings handle nonmonotonic sampled peak times; no global monotonicity assumption is made.
The fixed-X control uses X=0.1 on the SAME new physical grid and corrected boundary. It isolates the X formulation but is not identical to the historical200 m/floored production model: fixed-X diffusion depends on grid length. Historical calibrated n and errors are retained in comparison.csv. The selected target's sampling tolerance is not a parameter confidence interval.
Each trial extends its recession until residual flow is within0.1% of event excess, with two further days of buffer, or is explicitly marked censored after the extension limit. Censored/nonfinite/negative-flow/boundary-peak candidates do not qualify as valid best fits. Long recession tails can make high-n trials slow and/or invalid. Cache identities include input, observation and implementation hashes.

CALIBRATION SUMMARY
'''+summary.to_string()+'\n\nPER-NETWORK CANDIDATES\n'+results[['example','method','n_manning','abs_error_h','within_sampling','common_band_within_sampling','cap_fraction','normalized_shape_rmse']].sort_values(['example','method']).to_string(index=False)
    text+='''\n\nNUMERICAL VERIFICATION
21 implementation/regression tests passed in the river-hierarchy-rivgraph environment. This includes original preparation/engine tests, calibration tests, variable-X bounds, exact length and diffusion invariants, the first-substep boundary, root-only forcing and independent sparse/cascade agreement.
Full newly prepared Cunge files were routed by run_prepared_state for every parent. Every native reach flow is audited. The fast calibration recurrence is independently checked against the file-based sparse engine. Time halving and doubling the Cunge subsection count were checked at each fitted n. Peak-time comparisons retain the original forcing/report cadence; zero reported change does not imply zero sub-report-scale error.
Positive12 h and48 h rise pulses test every subsection at fitted n. Reported finite-window volume shortfalls include unresolved far tails; algebraic zero-frequency gain is also checked separately to distinguish tail truncation from loss of mass.

INDEPENDENT EVENTS
Two temporally nearest qualifying upstream peaks per network were selected from gauge records without using model performance. Detection:6-hour median,28-day peak separation and prominence>=15% of the upstream95th–5th percentile range. Windows span7 days before to14 days after the detection peak, exclude the calibration window plus7-day guard and do not overlap each other. Both gauges need at least50 samples and no gap/endpoint shortfall above6 hours. The raw upstream global maximum must lie within24 hours of the detection peak. Selection is frozen and audited in independent_events/.
28 events were selected across14 networks;15 and37 had no qualifying event. All fitted n, stored reference Q, geometry, grid and clock are inherited without refitting. Upstream observations are interpolated to the existing forcing cadence; the model keeps its held-forcing convention. First/last observation overlap defines the common interval. Events with nonpositive observed lag or peaks within one day of the window boundary are flagged unusable for the paired timing summary, not silently replaced with better-performing events.
This is automatically screened out-of-event evidence, not a final hydrological validation dataset. Multiple flood peaks, gauge/rating changes, tributary inflow and backwater can affect correspondence and amplitude; no lateral-flow correction is fitted. The seven-day antecedent window and steady first-flow initialization can also affect models with long memory. Normalized shape RMSE is evaluated at observed UTC without peak alignment and is diagnostic only. Paired summary medians use exactly the same usable events for both methods.

HYDRAULIC CONSISTENCY
hydraulic_consistency_diagnostic.csv compares frozen Q_reference with the wide-rectangular Manning discharge implied by prescribed B,h,S,n before celerity caps. Agreement is not forced. This exposes the limitations of independently prescribed depth, reference-flow allocation and calibrated effective roughness; it is not an observed-discharge error. In particular, example11 has large negative X and a highly diffusive reference parameterization. Its selected-event lag remains far below the observed63.5 h across a denser31-point n check. Spatial refinement does not repair that mismatch. Changing alpha to fit that event would confound numerical settings and hydraulics.
A separate uniform-reach continuous advection-diffusion/radiation calculation at the selected example11 n delays its peak by5.38 h relative to the Cunge recurrence and differs by about7.0% of inlet amplitude. Halving its FFT report step from150 to75 seconds changes that peak by75 seconds. Thus the independent diffusion benchmark still leaves most of the observed timing mismatch, while also showing that convergence within the Cunge recurrence is not proof of equivalence to a continuous diffusion solution. See diffusion_diagnostic_step150/75 outputs and cunge_diffusion_check.py; no hydraulic parameter was changed or fitted in this check.

QUANTITATIVE VERIFICATION AND TRANSFER SUMMARY
'''+json.dumps(conclusion,indent=2)
    text+='''\n\nINTERPRETATION
The new formulation is implemented and numerically testable, but calibration success alone does not establish it as a better physical model. Review the unresolved network11 mismatch, independent-event performance, wide-channel hydraulic-consistency diagnostics, and very long diffusion tails before replacing the existing default. A smaller n span is not established merely by achieving more fits inside a preferred band. There is still one fitted parameter per parent; alpha=2 remains a numerical choice.
No child-network runs or automatic promotion of these candidates were made. Per-parent cunge_prep/ directories are runnable inputs; cunge_run/ directories contain full sparse-engine outputs. The comparison control has matching fixed_x_prep/ files and a checked engine probe; its complete outlet series is produced by the independently verified native recurrence.

REPRODUCTION
python RAPID/investigations/cunge_calibration.py --workers 2
python RAPID/investigations/cunge_validation.py select
python RAPID/investigations/cunge_validation.py independent --workers 2
python RAPID/investigations/cunge_validation.py prepared --workers 2
python RAPID/investigations/cunge_report.py
The native calibration was executed in the geo environment; pure-Python optional dependencies were supplied from the installed base environment with an old-Plotly/NumPy compatibility shim. Regression tests, full prepared runs and independent-event checks used river-hierarchy-rivgraph. No environment installation was required.
'''
    for a,b in {'steps1':'steps 1','All16':'All 16','at0.05':'at 0.05','range0.010':'range 0.010',
                'A0.025':'A 0.025','within0.1%':'within 0.1%','Positive12':'Positive 12',
                'and48':'and 48','Detection:6':'Detection: 6',',28-day':', 28-day',
                'span7':'span 7','to14':'to 14','plus7':'plus 7','least50':'least 50',
                'above6':'above 6','within24':'within 24','across14':'across 14',
                ';15 and37':'; 15 and 37','example11':'example 11','observed63.5':'observed 63.5',
                'denser31':'denser 31','by5.38':'by 5.38','about7.0%':'about 7.0%',
                'from150 to75':'from 150 to 75'}.items():text=text.replace(a,b)
    (OUT/'REPORT.txt').write_text(text)
    print(json.dumps(conclusion,indent=2))


def figure_calibration(r,paths):
    fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    for method,color in [('fixed_x','#5473a4'),('cunge','#cf7443')]:
        g=r[r.method==method].sort_values('example');xx=np.arange(len(g))
        axs[0,0].plot(xx,g.n_manning,'o-',label=method,color=color)
        axs[0,1].plot(xx,np.maximum(g.abs_error_h*60,.01),'o-',label=method,color=color)
        axs[1,0].plot(xx,g.normalized_shape_rmse,'o-',label=method,color=color)
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xticks(xx,g.example);ax.set_xlabel('Network');ax.legend()
    axs[0,0].axhspan(.025,.06,color='gray',alpha=.12);axs[0,0].set_ylabel('Fitted Manning n');axs[0,0].set_title('One fitted parameter per network')
    axs[0,1].set(yscale='log',ylabel='Absolute timing error (minutes)',title='Calibration objective; floor 0.01 min for plotting')
    axs[1,0].set(ylabel='Normalized shape RMSE',title='Unfitted same-event diagnostic; no peak alignment')
    p=paths['11'];dense=pd.read_csv(p/'cunge_dense_diagnostic.csv')
    axs[1,1].plot(dense.n_manning,dense.lag_h,'o-',label='Cunge dense check');axs[1,1].axhline(63.5,color='black',ls='--',label='Observed lag')
    axs[1,1].set(xlabel='Manning n',ylabel='Peak lag (hours)',title='Network 11: unresolved hydraulic/model mismatch');axs[1,1].legend()
    fig.savefig(OUT/'calibration_comparison.png',dpi=170);plt.close(fig)


def figure_holdouts(p):
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,x,y,label in [(axs[0],p.abs_error_h_fixed,p.abs_error_h_cunge,'Absolute timing error (hours)'),
                         (axs[1],p.normalized_shape_rmse_fixed,p.normalized_shape_rmse_cunge,'Normalized shape RMSE')]:
        ax.scatter(x,y,color='#5473a4',s=35)
        limit=max(x.max(),y.max())*1.05;ax.plot([0,limit],[0,limit],color='grey',ls='--')
        ax.set(xlabel='Fixed X: '+label,ylabel='Cunge: '+label,xlim=(0,limit),ylim=(0,limit))
        for xx,yy,name in zip(x,y,p.event):
            threshold=10 if ax is axs[0] else .025
            if abs(xx-yy)>threshold or max(xx,yy)>.8*limit:
                near_edge=xx>.85*limit
                ax.annotate(name,(xx,yy),fontsize=8,xytext=(-5 if near_edge else 5,5),
                            ha='right' if near_edge else 'left',textcoords='offset points')
    fig.suptitle('Independent events: below diagonal favors Cunge; no refitting')
    fig.savefig(OUT/'independent_events.png',dpi=170);plt.close(fig)


def figure_hydrographs(results,paths):
    # Keep the figure generator shared with the review notebook.
    sys.path.insert(0,str(Path(__file__).parent))
    from cunge_hydrograph_figure import build_figure, save_figure
    figure=build_figure(results,paths)
    save_figure(figure)
    return figure


if __name__=='__main__':main()
