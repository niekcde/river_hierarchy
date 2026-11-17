"""Text report from completed study artifacts; never labels a partial run complete."""
import json
from pathlib import Path
import pandas as pd
import broad_study as b

def main():
 from plot_broad_comparison import main as plot_comparison
 from audit_broad_initial import main as audit_initial
 from status_broad import main as update_status
 import shutil
 plot_comparison()
 audit_initial();update_status()
 if (b.OUT/'investigation/current_mc/snapshot.json').exists():
  from audit_current_predictor import main as audit_current
  from plot_investigation37 import main as plot37
  audit_current();plot37()
  if (b.OUT/'investigation/37_junction_effect_comparison.json').exists():
   from summarize_hydraulic_investigation import main as summarize_investigation
   summarize_investigation()
 source=b.OUT/'source_snapshot';source.mkdir(exist_ok=True)
 for name in ['audit_broad_initial.py','check_combined_junction.py','plot_broad_comparison.py','status_broad.py','audit_current_predictor.py','investigate_junctions37.py','run_current_bed_controls.py','recover_hydraulic_metadata.py','plot_investigation37.py','summarize_hydraulic_investigation.py']:
  shutil.copy2(b.ROOT/'LISFLOOD_FP'/name,source/name)
 o=b.OUT;s=json.loads((o/'summary_status.json').read_text());inv=pd.read_csv(o/'inventory.csv',dtype={'example':str});cal=pd.read_csv(o/'calibration_summary.csv',dtype={'example':str});base=pd.read_csv(o/'baseline_summary.csv',dtype={'example':str});gr=pd.read_csv(o/'grid_summary.csv',dtype={'example':str});tr=pd.read_csv(o/'transfer_summary.csv',dtype={'example':str})
 lines=['BROAD LOCAL-INERTIAL ROUTING STUDY — 100 m working maximum','', 'COMPARISON CORRECTION (16 November): MC curves and mc_* columns below use the legacy October-13 reference, not the current fixed-Manning-depth MC setup used to train the accepted six-input predictor. The current MC also uses different collapsed bed profiles in five selected states. Do not interpret the legacy smaller-effect trend or example-37 sign reversal as a comparison against the current MC. See investigation/matched_current_mc_cases.csv and investigation/matched_current_mc_plotly.html for the matched centroid-window audit. Geometry-mismatched states are excluded from its strict comparison until rerun.','',f"Completed original attempts: {s['baselines_completed']}/15; calibration attempts: {s['calibrations_completed']}/15; timing-calibrated: {s['calibrated']}; selected collapse attempts: {s['transfers_attempted']}/30; accepted: {s['transfers_valid']}; fixed-roughness grid cases: {s['grid_cases_completed']}/63 (including reused 100 m originals).",'',
 'Scope and comparability',
 'All 15 original networks; two geometrically selected collapses per example, nearest approximately 5% and 30% reduction in physical reach count. Grid subset: 02, 15, 22, 37, 46, 48, 56 at maximum lengths 200, 100 and 50 m. Selection precedes model outcomes. Full production across 2,202 saved network states is deferred.',
 '100 m is a maximum cell length, not a minimum. Each physical reach is divided into ceil(L/100) equal cells. Short physical reaches remain intact. MC retains its own original computational segmentation; both methods use the same physical network, forcing, measured widths and saved slopes.',
 'The solver is the audited custom Python CPU translation of the supplied MATLAB local-inertial method. It is not the official LISFLOOD-FP executable and does not solve the advective momentum term. Prior example-56 MATLAB agreement applies to the adapted CPU MATLAB method with aligned forcing and normal-flow closure, not the unchanged supplied GPU script. This broader study runs Python. Generalized three-way connectivity passed an analytical symmetric steady-flow check.',
 'MC uses fixed width-proportional split fractions and reference hydraulic depths; its fitted n changes celerity at fixed depth. The local-inertial model solves evolving depths and pressure-driven interface flows, so the bifurcation shares can change. Calibration values need not coincide between these formulations, and differences in collapse response cannot be attributed to backwater alone without further flow-field diagnostics.',
 'The MC depth column is its stored reference hydraulic depth; the local-inertial maximum is the maximum evolving water depth across cells and time. These are different quantities and should not be treated as two observed flood-depth estimates.',
 'Widths and proxy bed elevations are fixed; slopes reconstructed from endpoints exactly match saved slopes and node elevations are consistent across all originals. Bed elevations remain the previously agreed water-surface proxies. The downstream boundary is evolving normal flow Q=B h^(5/3) sqrt(S)/n, using terminal physical slope. No outlet extension or 200 m floor is introduced in this experiment.',
 'Steady initial flow is solved until maximum node imbalance / max(inflow,1) <1e-8. Held one-minute upstream forcing is integrated to one-minute outlet means. The full buffered MC forcing horizon is used for final runs.',
 '', 'Metric definitions',
 'Collapse effects are child minus original; negative timing changes mean earlier passage. Peak timing uses a local quadratic interpolation around the global maximum, and is distinct from the reviewed calibration feature (especially the two-lobe midpoint in 54). The centroid uses positive excess above the common upstream baseline ramp between input-event start and end. Rising50/falling50 are crossing times at half each hydrograph’s peak excess above its initial discharge; they are not limb durations. All metrics use full one-minute output, not the reduced plotting data.',
 '', 'Calibration and numerical acceptance',
 'Only Manning n is fitted, uniformly per original network, within 0.01–0.10. Smaller-network scan: 0.01, 0.0175, 0.025, 0.04, 0.07, 0.10 and the original MC n; sign-changing valid timing brackets are refined. Unresolved targets near an accepted/rejected roughness frontier receive up to nine additional bisections and timing-bracket refinement. For expensive networks 22, 37, 41, 46 and 55, calibration instead reuses the original full run and starts in the timing-correction direction at 0.75 or 1.25 times the MC n, expanding by a factor of 1.5 only when needed within the same bounds. Existing completed trials are retained. This is a local timing search, not an exhaustive search for all possible peak switches. The reviewed downstream crest-time feature is retained, including the two-lobe midpoint for 54 and middle chronological equal maximum for 48. Features are evaluated within the observed downstream time coverage from the input-event start, allowing downstream peaks after the upstream event ends. Search runs omit unnecessary long tails; the selected n is rechecked over the full horizon.',
 'A timing fit requires error <=1 minute. Height is a secondary selector among timing-feasible candidates, not a fitted second parameter. Raw RMSE/NSE are reported separately over observation-supported intervals without baseline adjustment. Timing agreement is not whole-hydrograph or independent validation. The selected n is frozen for the two collapses; transfers from an unresolved parent fit remain diagnostic.',
 'CFL starts at 0.35, with 0.175 and 0.0875 retries if needed. Event runs must have no active critical-flow cap, <1e-6 m3 depth-clipping addition and relative volume residual <1e-9. A cap-active steady initial state is recorded as outside this strict verified regime, without redundant time-step retries. A limiter rejection is not proof that the supplied capped algorithm is universally invalid; it prevents silently accepting a regime not validated by the prior study. Reverse internal discharge is diagnosed, not automatically rejected.',
 'Fitted original/collapse comparisons use the smallest accepted CFL required by that example: successful higher-CFL runs are repeated at this common setting, and the original timing fit is rechecked. Fixed-MC-roughness mesh tables retain their recorded CFL; differing CFL values must be recognized as combined space/time-step sensitivity, not pure spatial convergence.',
 'Tail settling is checked against 0.1% of the upstream flow range over the last 48 hours. Unsettled cases must not support centroid conclusions until their tails are extended.',
 '', 'Calibration results',cal[[x for x in ['example','status','mc_n','selected_n','timing_error_minutes','feature_kind','height_distance_cms','raw_downstream_nse','mc_stored_reference_depth_max_m','maximum_depth_m','seconds'] if x in cal]].to_string(index=False),
 '', 'Original runs at MC roughness',base[[x for x in ['example','valid','n','timing_error_minutes','maximum_depth_m','maximum_face_froude','seconds'] if x in base]].to_string(index=False),
 '', 'Grid comparison: differences in collapse effects, 100 m minus 50 m (minutes)']
 diffs=[]
 for (ex,state),g in gr[gr.state.ne('S000_base')].groupby(['example','state']):
  g=g.set_index('length')
  if 100 not in g.index or 50 not in g.index or not(g.loc[100,'valid'] and g.loc[50,'valid']):continue
  r=dict(example=ex,state=state)
  for k in ['peak','centroid','rising50','falling50']:
   col=k+'_change_minutes';r[k]=g.loc[100,col]-g.loc[50,col]
  diffs.append(r)
 df=pd.DataFrame(diffs);df.to_csv(o/'grid_effect_differences_100_minus_50.csv',index=False);lines.append(df.to_string(index=False))
 lines.extend(['','Grid differences are numerical-sensitivity diagnostics at fixed MC roughness. They do not prove asymptotic convergence for every fitted n. Small collapse signals can have large relative uncertainty even when absolute differences are small. The 100 m working choice is retained as requested; failures and larger sensitivities must remain visible before the full production run.','',f"Unsettled final tails: {len(s['unsettled_tails'])}"])
 shape_cols=['example','status','local_minus_mc_peak_height_percent','local_minus_mc_peak_minutes','local_minus_mc_centroid_minutes','local_minus_mc_rising50_minutes','local_minus_mc_falling50_minutes']
 lines.extend(['','Selected-roughness original versus MC reference',cal[[c for c in shape_cols if c in cal]].to_string(index=False),'Timing differences in this table are local inertia minus MC, in minutes. Both models retain their own reference parameterization; half-height crossings also reflect differences in hydrograph shape and height.'])
 effect_path=o/'collapse_effects_summary.csv'
 if effect_path.exists():
  effect=pd.read_csv(effect_path,dtype={'example':str});cols=['example','state','parent_calibration_status','peak_change_minutes','mc_peak_change_minutes','centroid_change_minutes','mc_centroid_change_minutes','rising50_change_minutes','mc_rising50_change_minutes','falling50_change_minutes','mc_falling50_change_minutes']
  lines.extend(['','Selected collapse effects versus MC',effect[[c for c in cols if c in effect]].to_string(index=False),'Missing half-height crossings are left undefined. Unresolved parent calibrations are diagnostic comparisons, not successful calibrated transfers.'])
 junction=o/'combined_junction_check.json'
 if junction.exists():
  jr=json.loads(junction.read_text());lines.extend(['','Combined merge/split junction diagnostic', 'Original networks 15, 22 and 37 contain simultaneous merge/split nodes, absent from 56. An independent constant-width uniform-Manning test compares separate parallel middle nodes with a shared 2-in/2-out node. Both versions start with mass-balanced interface fluxes. The separate-node control preserves 2 m depth to roundoff. With the inherited pairwise min-width interfaces, the shared-node case departs from that ideal uniform equilibrium; the final maximum depth departure after 24 hours is listed below. It decreases approximately linearly with cell length. This exposes a finite-grid junction bias, not a mass-conservation failure or a proven explanation of the real-network limiter rejections. The identified capped interfaces are elsewhere. No junction rule was changed in the study.',pd.DataFrame([dict(shared_middle_node=r['shared_middle_node'],max_length_m=r['max_length_m'],cfl=r['cfl'],maximum_final_depth_departure_m=r['maximum_final_depth_departure_m']) for r in jr['runs']]).to_string(index=False)])
 fp=o/'runtime_forecast_examples.csv'
 fitted=o/'calibrated_grid_checks.csv'
 if fitted.exists():
  lines.extend(['','Targeted 100 m versus 50 m checks at fitted roughness',pd.read_csv(fitted,dtype={'example':str}).to_string(index=False),'Original timing shifts and shifts in collapse effects are reported separately. The selected checks cover 02, 03, 15, 37 and 56; only completed checks appear in this table. They are not a fitted-roughness mesh study of all 15 networks.'])
 if fp.exists():
  rt=pd.read_csv(fp,dtype={'example':str});lines.extend(['','Runtime forecast',rt.to_string(index=False),'Forecast scales measured full-run wall time by cell count and an initial normal-depth Courant-rate estimate for each saved network. Times include startup, steady initialization and contention from concurrent jobs. This is an approximate serial-equivalent workload, not an elapsed-time promise. Different depths, rejected regimes, time-step retries and thermal/load changes can alter costs materially. Cases without an accepted reference are omitted and must not be interpreted as zero cost.'])
 lines.extend(['','Artifacts','hydrographs_plotly.html: raw observed downstream plus the frozen prior display-offset curve, MC original and selected collapses, local-inertial original at MC roughness and selected roughness, and fixed-n selected collapses. Select the example with the menu. Missing rejected runs are not plotted. Model curves use per-10-minute minima/maxima for display; metrics use full one-minute data.','grid_sensitivity_plotly.html: peak, centroid and half-height rising/falling timing effects versus maximum cell length. Missing points indicate rejected or unfinished configurations.','collapse_comparison_plotly.html: cross-model scatterplots of selected collapse effects for peak, centroid and half-height crossings; unresolved parent fits are shown separately.',
 'CSV tables retain timings, depths, Froude numbers, balance diagnostics, fit status, selected geometry and runtime. Per-trial result.json and wave.parquet retain numerical records.'])
 limiter=o/'investigation/limiter_retained_48_fit.json'
 if limiter.exists():
  lr=json.loads(limiter.read_text());selected=lr['selected']
  lines.insert(3,f'CALIBRATION INTERPRETATION: unresolved statuses below are conditional on the strict no-limiter-activation gate. The supplied MATLAB scheme includes a critical-flow limiter. A separate limiter-retained example-48 diagnostic fits timing to {selected["timing_error_minutes"]:.1f} min at 100 m, also reproduced at half CFL, but 50 m checks produce negative-depth clipping and are not accepted. Thus the 53.5-minute uncapped offset is not a proven intrinsic calibration limit; the cap-active numerical regime needs further verification. See investigation/limiter_retained_48_fit.json.')
 (o/'report.txt').write_text('\n\n'.join(lines)+'\n')
 print(o/'report.txt')
if __name__=='__main__':main()
