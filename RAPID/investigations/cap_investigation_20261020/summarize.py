"""Check preservation and record diagnostic conclusions in a new notebook."""
import json
import nbformat as nb
import pandas as pd
from audit import OUT,EXT,ROOT,read,save,sha
from rapid_tools.event_window_batch import settling_hours

def main():
 d=pd.read_csv(OUT/'routing/metrics.csv',dtype={'example':str})
 plan=read(EXT/'routing_plan.json')
 for case in plan['cases']:
  ex=case['example'];wave=pd.read_parquet(OUT/f'routing/{ex}.parquet')
  for idx,row in d[d.example.eq(ex)].iterrows():
   required,settled=settling_hours(wave.time_utc,wave[row.state+'__'+row.scenario].to_numpy(),case['window']['end_utc'],case['final_cms'],case['tolerance_cms'])
   d.loc[idx,'required_tail_hours']=required;d.loc[idx,'settled']=settled
 d.to_csv(OUT/'routing/metrics.csv',index=False)
 originals=read(EXT/'state_metrics.parquet')
 controls=d[d.scenario.eq('capped_control')]
 differences=[]
 for row in controls.itertuples():
  original=originals[(originals.example==row.example)&originals.state.eq(row.state)&originals.window.eq('0.5%')].iloc[0]
  # Absolute centroid locations assess grid effects without introducing a new response definition.
  differences.append(dict(example=row.example,state=row.state,max_discharge_difference_cms=row.max_absolute_difference_from_published_cms))
 pd.DataFrame(differences).to_csv(OUT/'grid_controls.csv',index=False)
 snapshots=[OUT/'protected_before.json',OUT/'protected_active_calibration_before.json',OUT/'protected_extension_before.json',OUT/'source_hashes.json',OUT/'conditional_recalibration/sources.json',OUT/'routing/sources.json']
 protected={}
 for path in snapshots:
  for p,h in json.loads(path.read_text()).items():
   if p in protected:assert protected[p]==h,p
   protected[p]=h
 changed=[p for p,h in protected.items() if sha(p)!=h]
 save(OUT/'preservation_check.json',dict(files_checked=len(protected),changed=changed,passed=not changed))
 assert not changed,changed
 audit=pd.read_csv(OUT/'state_cap_audit.csv',dtype={'example':str})
 child=d[d.state.eq('all_closed_units_collapsed')]
 table=child.pivot(index='example',columns='scenario',values='centroid_shift_hours')
 table['uncapping_effect_hours']=table.uncapped-table.capped_control
 table.to_csv(OUT/'segmented_centroid_summary.csv')
 summary=dict(states_audited=len(audit),states_with_active_upper_cap=int(audit.upper_capped_links.gt(0).sum()),networks_with_active_cap=sorted(audit.loc[audit.upper_capped_links.gt(0),'example'].unique()),routed_scenarios=len(d),all_routes_settled=bool(d.settled.all()),minimum_native_flow_cms=float(d.minimum_native_flow_cms.min()),baseline_changed=bool((d.baseline_cms!=d.published_baseline_cms).any()),preserved_files=len(protected),scope='All saved states audited analytically and for grid preparation; actual rerouting limited to 15 parent/segmented pairs. Conditional recalibration varies n only, with beta fixed. No replacement production outputs.')
 save(OUT/'summary.json',summary)
 notebook=nb.v4.new_notebook()
 notebook.cells=[
 nb.v4.new_markdown_cell("""# Upper-celerity cap investigation — 20 October 2026

**Separate diagnostics only. All pre-existing results and active calibration files remain unchanged.**

We audit the 5 m/s upper cap across all saved states, then reroute the parent and segmented collapse for each of the 15 examples. Three scenarios use a common refined grid and timestep within each example: capped control; uncapped at current calibration; uncapped with conditional roughness recalibration. The lower celerity bound is retained.

The uncapped ceiling is above every raw celerity throughout the declared roughness range. Physical link lengths and geometry are preserved. Grid refinement is necessary where removing the cap violates the existing spatial bound; the matching capped control isolates this numerical change.

The conditional calibration adjusts **only Manning n**, preserving beta, the archived calibration forcing, targets and crest operator. It is not a full joint calibration or a validated replacement model. Example 54 retains its archived calibration operator; response peak timings still use global maxima.

All response comparisons use the existing shared 0.5% analysis-window boundaries. A single baseline is shared across diagnostic states and scenarios; any required baseline change is explicitly reported below. Example 54's finite-window distribution includes its transition to a higher ending equilibrium."""),
 nb.v4.new_code_cell("""from pathlib import Path
import json
import pandas as pd
import plotly.express as px
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'RAPID').exists())
out = root/'RAPID/outputs/cap_investigation_20261020'
display(json.loads((out/'summary.json').read_text()))
display(json.loads((out/'preservation_check.json').read_text()))"""),
 nb.v4.new_markdown_cell("""## Where the cap is active

The audit's `change_in_collapse_shift_hours` is an analytic mean-route-time contrast, **not a simulated centroid shift**. It is useful for screening; actual hydrograph responses appear separately below."""),
 nb.v4.new_code_cell("""audit = pd.read_csv(out/'network_cap_audit.csv', dtype={'example':str})
display(audit)
routes = pd.read_csv(out/'routing/metrics.csv', dtype={'example':str})
summary = pd.read_csv(out/'segmented_centroid_summary.csv', dtype={'example':str})
display(summary)
px.bar(summary, x='example', y='uncapping_effect_hours',
       labels={'uncapping_effect_hours':'Uncapped − capped centroid shift (hours)'},
       title='Effect of removing the cap at unchanged calibration').show()"""),
 nb.v4.new_markdown_cell("""## Conditional recalibration

The original timing target can be restored by changing roughness in the affected parents. This does not imply unchanged collapse responses: both parent and child receive the recalibrated roughness, and their geometry differs. Peak-height interval distances are reported because recovering timing alone may worsen magnitude agreement."""),
 nb.v4.new_code_cell("""records = json.loads((out/'conditional_recalibration/results.json').read_text())
display(pd.DataFrame([{
    'example':r['example'], 'old_n':r['old_n'], 'new_n':r['new_n'],
    'uncapped_old_n_timing_error_minutes':r['uncapped_old_n']['timing_error_minutes'],
    'refitted_timing_error_minutes':r['selected']['timing_error_minutes'],
    'old_height_distance_cms':r['old_calibration_height_distance_cms'],
    'new_height_distance_cms':r['selected']['height_distance_cms']
} for r in records if r['refitted']]))
display(routes[['example','state','scenario','centroid_shift_hours','peak_shift_hours',
                'minimum_native_flow_cms','settled','max_absolute_difference_from_published_cms']])"""),
 nb.v4.new_markdown_cell("""## Interpretation and limits

The upper cap is not negligible under the current calibration, especially for example 22. Stable, nonnegative uncapped routes on appropriately refined grids demonstrate numerical feasibility for the tested parent/segmented states; they do not establish physical credibility of raw celerities approaching 13 m/s, or guarantee every partial collapse will route acceptably.

Example 22's segmented centroid response changes from approximately −5.26 h to −8.24 h at fixed calibration and −8.93 h after conditional timing recalibration. Restoring parent timing does not restore the original collapse response. Other networks are substantially less sensitive.

Keep the frozen production results unchanged. Before choosing an uncapped replacement, independently assess the prescribed depth, slope and roughness assumptions, and perform a complete, consistent calibration and response regeneration. Do not remove the cap solely because it moves one case closer to a predictor's estimate. The remaining peak differences between matching-grid controls and published curves should also be distinguished from the much more stable centroid.

This investigation does not refit the forest, alter predictors, or assess its predictive scores on changed targets.""")]
 notebook.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
 nb.write(notebook,ROOT/'RAPID/notebooks/hydrograph_cap_investigation_20261020.ipynb')
 print(json.dumps(summary,indent=2));print(table.to_string())

if __name__=='__main__':main()
