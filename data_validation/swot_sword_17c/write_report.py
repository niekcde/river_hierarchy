from pathlib import Path
import json
import pandas as pd
D=Path(__file__).resolve().parent
m=json.loads((D/'run_metadata.json').read_text()); s=pd.read_csv(D/'example_summary.csv')
a=m['all_examples']; b=m['generated_examples']
lines=['# SWOT versus SWORD 17c node validation','',
'Filtered local SWOT does **not** provide more nodes with a valid WSE median overall. SWORD has fewer missing medians in both the 17 saved example corridors and the 16 corridors with generated networks.','',
'| Scope | SWORD nodes | SWORD valid p50 | SWOT valid p50 | SWORD missing | SWOT missing | Recovered SWORD gaps | Net SWOT gain |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,d in [('17 saved example corridors',a),('16 generated-network example corridors',b)]:
 lines.append('| '+name+' | '+' | '.join(f'{d[k]:,}' for k in ['sword_nodes','sword_valid','swot_valid','sword_missing','swot_missing','recovered_by_swot','net_gain'])+' |')
lines += ['',f'For the 17 corridors, SWORD coverage is {100*a["sword_valid"]/a["sword_nodes"]:.2f}% and SWOT coverage is {100*a["swot_valid"]/a["sword_nodes"]:.2f}%. SWOT recovers 25 SWORD gaps, but 182 nodes with a SWORD median have no filtered SWOT median; 880 nodes lack both. No SWOT node/reach IDs on these reaches are absent from SWORD 17c.', '',
'## Inputs and method','',
'- SWORD: final `/Volumes/PhD/SWORD/v17c/sword_{AS,NA,SA}_v17c_nodes.parquet`, not beta. The requested median is named `wse_obs_p50` in these files. Missing means null/nonfinite or a large negative fill code; valid negative elevations are retained.',
'- SWOT: all 17,963 non-hidden node Parquet files in `/Volumes/PhD/SWOT/RiverSP_D_parq/node`, including all geographic filename codes and product versions. All files were read successfully; 564,049,068 archive rows were accounted for. Reach products were not mixed into a node-level statistic. Files beginning `._` are filesystem sidecars, not datasets.',
'- Example membership: all nodes on `reaches_between` plus upstream/downstream endpoint reaches in `SWORD_gauge_match/outputs/hierarchy_example_reach_summary.csv`. All requested reaches exist in SWORD. No extra reach buffer, bounding-box crop, or nearest-node matching. Exact `(node_id, reach_id)` joins preserve absent observations. Example 10 is the only saved corridor without a generated network.',
'- The broader station registry also mentions examples 6, 9, 12, 14, 47 and 52, which have neither saved corridors in that summary nor generated network roots. They are outside these reported corridor results; the report does not assert validation of those older station-only entries.',
'- Existing filter found in the sibling repository `river-morphology-atlas/open_SWOT_files.py`, also used by its SWOT diagnostic notebooks: `wse_sm_u <= 1`, `dark_frac <= 0.5`, `10000 <= abs(xtrk_dist) <= 60000`, `wse_sm > 0`. No additional node-quality flag or minimum-observation count is imposed by that code. The source module was inspected but not executed, because it also deletes hidden files.',
'- From 1,010,146 observations on the selected reaches, 473,247 pass that predicate. SWOT p50 is the ordinary median of passing `wse_sm` rows per exact node/reach pair, matching the repository aggregation. Source observations span 2023-07-28 to 2025-11-12 after filtering. There is no WSE interpolation, smoothing after aggregation, or fallback to another SWORD WSE field.', '',
'## Per example','',
'| Example | Nodes | SWORD valid | SWOT valid | SWORD missing | SWOT missing | Recovered gaps | Net SWOT gain |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in s.itertuples():
 lines.append('| '+str(r.example_id)+(' (no generated network)' if not r.generated_network else '')+' | '+' | '.join(f'{getattr(r,k):,}' for k in ['sword_nodes','sword_valid','swot_valid','sword_missing','swot_missing','recovered_by_swot','net_gain'])+' |')
lines += ['',
'Only examples 41 and 55 have a net coverage increase (+1 node each). Recovered gaps occur in examples 22 (8), 41 (1), 46 (8), 54 (6), and 55 (2).','',
'## Filter and duplicate sensitivities','',
'The existing filter allows negative missing-value codes in uncertainty/dark-fraction fields because it only applies upper bounds. Sixteen otherwise passing rows have invalid QC values. Requiring finite fields and nonnegative uncertainty/dark fraction removes the sole SWOT support at node **44250000060371** (example 54). Therefore strict QC gives **18,184** valid SWOT medians for all 17 corridors (1,063 missing; 24 recovered gaps; net -158), or **16,863** for the generated 16 (1,008 missing; net -140). The overall conclusion is unchanged. The original repository predicate remains the primary reported comparison.', '',
'The passing observations contain 34,456 repeated rows by `(node_id, reach_id, cycle, pass, time_str)`, in 30,994 groups; 5,962 groups contain differing WSE values. These include overlapping files/product versions. The primary p50 intentionally retains all file rows as the repository does. A diagnostic retaining the first sorted-file row per observation changes 10,219 node medians, with median absolute change 0.004345 m and maximum 4.1028 m; it leaves coverage unchanged. This sensitivity is not a scientifically preferred product-version selection. Any subsequent hydraulic use should choose a product/version policy explicitly.', '',
'Among nodes with both medians, the median SWOT-minus-SWORD difference is -0.08533 m and median absolute difference is 0.236825 m. Example-level differences are in `example_summary.csv`; some exceed 2 m. These are agreement diagnostics, not accuracy estimates: the SWORD aggregation dates/filter are not established here, and the local SWOT archive combines product versions.', '',
'## Deliverables and verification','',
'- `example_summary.csv`: per-example coverage and WSE agreement.',
'- `example_node_comparison.csv`: node-level p50, observation count, time range and coverage category, with example IDs.',
'- `unique_node_comparison.csv`: the unique SWORD node universe and both medians, including the strict-QC alternative.',
'- `target_observations.parquet`: all selected-reach source observations and filter flags.',
'- `file_manifest.csv`, `example_reaches.csv`, `run_metadata.json`: source inventory, exact scope, hashes, counts and filter provenance.',
'- `duplicate_observation_groups.csv`, `duplicate_sensitivity.csv`, `invalid_qc_rows.csv`, `invalid_qc_coverage_sensitivity.csv`: diagnostic detail.',
'- `verification.json`: passed all file/coverage accounting, uniqueness and reach-completeness checks; independently re-read a complete source granule without predicate pushdown and recomputed 100 medians with NumPy.', '',
'Reproduce the primary comparison and verification from the repository root:', '',
'```sh','/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python data_validation/swot_sword_17c/validate.py','/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python data_validation/swot_sword_17c/verify.py','```','',
'Source datasets and production hydraulic outputs were not modified.']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
