"""Read-only source audit of delays; writes only a separate diagnostic directory.

Run with the river-hierarchy-rivgraph Python environment. No fits or routing runs.
The initial population is the frozen 2,185-case baseline, excluding combinations.
"""
from pathlib import Path
import hashlib
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'RAPID/outputs/uncapped_event_window_20261020'
DEST = OUT / 'delay_side_project_20261026'
EXT = ROOT / 'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
sys.path.insert(0, str(ROOT / 'RAPID/src'))
sys.path.insert(0, str(ROOT))
from rapid_tools.cunge import topology

SOURCES = {}


def read(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == '.json':
        return json.loads(path.read_text())
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, dtype={'example': str})


def centroid(t, q):
    """Exact first moment of the piecewise-linear excess discharge."""
    x = t - t[0]
    dt = np.diff(x)
    volume = np.sum(dt * (q[:-1] + q[1:]) / 2)
    moment = np.sum(dt / 6 * (x[:-1] * q[:-1] +
                    4 * (x[:-1] + dt / 2) * (q[:-1] + q[1:]) / 2 + x[1:] * q[1:]))
    assert volume > 0
    return t[0] + moment / volume, volume


def routing_summary(frame):
    """Network transfer mean, path mixing variance, and Cunge variance in hours."""
    order, edges, root, end = topology(frame)
    p, m1, m2, diffusion = {root: 1.}, {root: 0.}, {root: 0.}, {root: 0.}
    for u in order:
        for i in edges.get(u, []):
            r = frame.iloc[i]
            w, v = r.split_fraction, r.v
            mu = r['count'] * r.rapid_k / 3600
            variance = r['count'] * r.rapid_k**2 * (1 - 2*r.rapid_x) / 3600**2
            p[v] = p.get(v, 0) + p[u] * w
            m1[v] = m1.get(v, 0) + w * (m1[u] + p[u] * mu)
            m2[v] = m2.get(v, 0) + w * (m2[u] + 2*mu*m1[u] + p[u]*mu**2)
            diffusion[v] = diffusion.get(v, 0) + w * (diffusion[u] + p[u]*variance)
    np.testing.assert_allclose(p[end], 1, atol=1e-10)
    np.testing.assert_allclose(m1[end], (frame.reference_flow_fraction * frame.link_length_m /
                                        frame.rapid_celerity_mps).sum()/3600, rtol=1e-10)
    # MultiDiGraph preserves parallel reaches when counting splits and joins.
    graph = nx.MultiDiGraph()
    graph.add_edges_from(zip(frame.u, frame.v))
    return dict(mean_h=m1[end], path_variance_h2=m2[end]-m1[end]**2,
                diffusion_variance_h2=diffusion[end], links=len(frame), nodes=len(graph),
                splits=sum(graph.out_degree(u)>1 for u in graph),
                joins=sum(graph.in_degree(u)>1 for u in graph),
                cycle_rank=len(frame)-len(graph)+1, cells=int(frame['count'].sum()),
                k_min_s=frame.rapid_k.min(), k_max_s=frame.rapid_k.max(),
                x_min=frame.rapid_x.min(), x_max=frame.rapid_x.max(),
                capped_links=int(frame.rapid_celerity_capped.sum()))


def main():
    DEST.mkdir(exist_ok=True)
    records = read(OUT / 'validation_group_inventory_20261023/records.parquet')
    assert len(records) == 2185
    errors = read(OUT / 'frozen_failure_audit_20261026/all_errors.parquet')
    d = errors.merge(records[['example', 'state', 'edit_type']], on=['example', 'state'], validate='many_to_one')
    assert len(d) == 4370
    d['observed_minutes'] = d.centroid_shift_hours * 60
    d['predicted_minutes'] = d.predicted_hours * 60
    d['observed_class'] = np.select([d.observed_minutes > 1, d.observed_minutes < -1], ['delay', 'advance'], default='near_zero')
    d['predicted_class'] = np.select([d.predicted_minutes > 1, d.predicted_minutes < -1], ['delay', 'advance'], default='near_zero')
    d.to_parquet(DEST / 'case_inventory.parquet', index=False)
    counts = records.assign(positive=records.centroid_shift_hours>0,
                            subminute=(records.centroid_shift_hours>0)&(records.centroid_shift_hours<1/60),
                            over_minute=records.centroid_shift_hours>1/60,
                            exact_minute=records.centroid_shift_hours.eq(1/60))
    counts = counts.groupby('edit_type').agg(total=('state','size'), positive=('positive','sum'),
                    subminute=('subminute','sum'), over_minute=('over_minute','sum'), exact_minute=('exact_minute','sum'))
    counts.to_csv(DEST / 'delay_counts.csv')
    d.groupby(['model','edit_type','observed_class','predicted_class']).size().rename('count').to_csv(DEST/'classification_counts.csv')
    # Both simple delays; largest missed delay, strongest false alarm, and an
    # accurately predicted substantial delay for each island/member category.
    depth = d[d.model.eq('depth_informed_10')].copy()
    selected = []
    for _, row in depth[(depth.edit_type=='Simple') & (depth.observed_minutes>1)].iterrows():
        selected.append(dict(example=row.example, state=row.state, reason='simple_centroid_delay', anchor=''))
    for kind in ['Island', 'Compound member']:
        f = depth[depth.edit_type.eq(kind)]
        pools = {
            'successful_delay': f[(f.observed_minutes>5)&(f.predicted_minutes>1)].sort_values('abs_error_hours'),
            'missed_delay': f[(f.observed_minutes>1)&(f.predicted_minutes<=1)].sort_values('observed_minutes', ascending=False),
            'false_delay': f[(f.observed_minutes < -1)&(f.predicted_minutes>1)].sort_values('predicted_minutes', ascending=False),
        }
        for reason, pool in pools.items():
            if pool.empty:
                continue
            row = pool.iloc[0]
            anchor = f'{row.example}/{row.state}'
            selected.append(dict(example=row.example, state=row.state, reason=reason, anchor=anchor))
            # Same-network/type advance, nearest in log(parent route length,
            # parent width). Matching uses parent properties, not child outcomes.
            controls = f[(f.example==row.example)&(f.observed_minutes < -1)&(f.state!=row.state)].copy()
            if not controls.empty:
                cols = ['B_parent_route_length_mean', 'B_parent_width_mean']
                assert (controls[cols]>0).all().all() and (row[cols]>0).all()
                controls['distance'] = np.sqrt(((np.log(controls[cols].astype(float))-
                                                 np.log(row[cols].astype(float)))**2).sum(axis=1))
                control = controls.sort_values(['distance','state']).iloc[0]
                selected.append(dict(example=control.example,state=control.state,reason='matched_advance',anchor=anchor,
                                     match_distance=control.distance))
    selection = pd.DataFrame(selected)
    selection.merge(depth, on=['example','state'], validate='many_to_one').to_csv(DEST/'selected_cases.csv', index=False)
    jobs = {}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json', EXT/'routing_plan.json', OUT/'routing_plan.json']:
        for case in read(path)['cases']:
            for job in case['jobs']:
                for name in {job['state_id'], *job.get('aliases', [])}:
                    jobs[case['example'], name] = job
    old, new = read(EXT/'state_metrics.parquet'), read(OUT/'review/uncapped_state_metrics.parquet')
    states = pd.concat([old[~old.example.isin(new.example)], new])
    states = states[states.window.eq('0.5%')].set_index(['example','state'])
    results, links, waves = [], [], []
    for ex, state in selection[['example','state']].drop_duplicates().itertuples(index=False, name=None):
        row = depth[(depth.example==ex)&(depth.state==state)].iloc[0]
        frames = {side: read(jobs[ex,name]['parameters']) for side,name in [('parent','S000_base'),('child',state)]}
        result = dict(example=ex,state=state,edit_type=row.edit_type,observed_min=row.observed_minutes,
                      predicted_min=row.predicted_minutes,peak_min=row.peak_shift_hours*60)
        for side, frame in frames.items():
            result.update({f'{side}_{k}': v for k,v in routing_summary(frame).items()})
        result['transport_min'] = (result['child_mean_h']-result['parent_mean_h'])*60
        parent, child = [frames[side].set_index('id_link') for side in ['parent','child']]
        added, removed = child.index.difference(parent.index), parent.index.difference(child.index)
        shared = parent.index.intersection(child.index)
        changed = shared[(parent.loc[shared,['u','v']].to_numpy()!=child.loc[shared,['u','v']].to_numpy()).any(axis=1)]
        result.update(added_link_ids=','.join(map(str,added)), removed_link_ids=','.join(map(str,removed)),
                      reconnected_link_ids=','.join(map(str,changed)))
        for side, frame in [('parent',parent),('child',child)]:
            f = frame.copy()
            f['side'], f['example'], f['state'] = side, ex, state
            f['transport_contribution_min'] = f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps/60
            f['membership'] = np.where(f.index.isin(shared), 'shared', 'removed' if side=='parent' else 'added')
            links.append(f.reset_index())
        pr, cr = states.loc[(ex,'S000_base')], states.loc[(ex,state)]
        for field in ['rise_seconds','fall_seconds','baseline_cms']:
            np.testing.assert_allclose(pr[field],cr[field],rtol=0,atol=1e-7)
        a,b,base = pr.rise_seconds,pr.fall_seconds,pr.baseline_cms
        data = {}
        for side, metric in [('parent',pr),('child',cr)]:
            f = read(metric.outlet)
            assert SOURCES[str(Path(metric.outlet))]==metric.outlet_sha256
            data[side] = (f.time_utc.astype('int64').to_numpy()/1e9, f.q_outlet_cms.to_numpy())
        tp,qp = data['parent']; tc,qc = data['child']; shift = result['transport_min']*60
        t = np.unique(np.r_[a,b,tp[(tp>a)&(tp<b)],tc[(tc>a)&(tc<b)],(tp+shift)[(tp+shift>a)&(tp+shift<b)]])
        pq,cq,sq = np.interp(t,tp,qp)-base,np.interp(t,tc,qc)-base,np.interp(t-shift,tp,qp)-base
        pm,pv=centroid(t,pq); cm,cv=centroid(t,cq); sm,sv=centroid(t,sq)
        np.testing.assert_allclose((cm-pm)/60,row.observed_minutes,atol=1e-6)
        result.update(translation_min=(sm-pm)/60,window_correction_min=(sm-pm)/60-result['transport_min'],
                      shape_residual_min=(cm-sm)/60,volume_change_pct=100*(cv/pv-1),
                      translation_padded=bool(t[0]-shift<tp[0] or t[-1]-shift>tp[-1]),
                      window_start_excess=pq[0],window_end_excess=pq[-1])
        np.testing.assert_allclose(result['transport_min']+result['window_correction_min']+result['shape_residual_min'],row.observed_minutes,atol=1e-6)
        results.append(result)
        waves.append(pd.DataFrame(dict(example=ex,state=state,hours=(t-a)/3600,parent=pq,child=cq,translated_parent=sq)))
    result = pd.DataFrame(results)
    result.to_csv(DEST/'mechanisms.csv',index=False)
    pd.concat(links).to_parquet(DEST/'routing_links.parquet',index=False)
    pd.concat(waves).to_parquet(DEST/'waveforms.parquet',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(waves),1,figsize=(10,2.5*len(waves)),constrained_layout=True)
    for ax,wave in zip(np.atleast_1d(axes),waves):
        for col,color in [('parent','#555555'),('child','#0072B2'),('translated_parent','#D55E00')]:
            ax.plot(wave.hours,wave[col],label=col,color=color,linewidth=1)
        ax.set_title(f'{wave.example.iloc[0]} / {wave.state.iloc[0]}',fontsize=10)
        ax.set_xlabel('Hours from shared window start'); ax.set_ylabel('Excess Q (m³/s)'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(DEST/'hydrographs.png',dpi=120);plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,6),constrained_layout=True)
    for axis_row,wave in zip(axes,waves[:2]):
        for col,color in [('parent','#555555'),('child','#0072B2'),('translated_parent','#D55E00')]:
            axis_row[0].plot(wave.hours,wave[col],label=col,color=color,linewidth=1.2)
        axis_row[1].plot(wave.hours,wave.child-wave.parent,label='Child − parent',color='#0072B2')
        axis_row[1].plot(wave.hours,wave.child-wave.translated_parent,label='Child − translated parent',color='#D55E00')
        axis_row[1].axhline(0,color='grey',linewidth=.6)
        for ax in axis_row:
            ax.set_title(f'{wave.example.iloc[0]} / {wave.state.iloc[0]}')
            ax.set_xlabel('Hours from shared window start');ax.set_ylabel('Excess Q / difference (m³/s)')
            ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(DEST/'simple_delays.png',dpi=160);plt.close(fig)
    def table(frame):
        columns=list(frame.columns)
        lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for values in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in values)+' |')
        return '\n'.join(lines)
    simple=result[result.edit_type.eq('Simple')]
    report='''# Delay side project: first diagnostic pass, 2026-10-26

Population: frozen baseline of 2,170 regular cases and 15 segmented cases.
New combination experiments are excluded. Positive means child minus parent.
Sub-minute means strictly 0 < centroid shift < 1 minute; no exact one-minute rows.
These are calculated positive shifts, not a claim of resolved physical significance.

## Counts

'''+table(counts.reset_index())+'''

There are 182 positive sub-minute shifts, all islands or compound members;
six are below one second (smallest about 0.46 seconds). There are 130 delays
above one minute, hence 312 positive shifts in total. The 15 segmented cases
have no positive centroid shifts.

## Two simple-unit delays

All entries in this accounting table are minutes. Transport comes from the
saved network transfer mean; window correction is the effect of measuring
a purely translated parent waveform in the original window. The shape term
is actual child minus translated parent, measured with the same centroid operator.
These terms sum to the observed shift; the shape term is not uniquely diffusion.

'''+table(simple[['example','state','observed_min','predicted_min','transport_min','window_correction_min','shape_residual_min']])+'''

Both cases become faster in mean transport. A translated parent would advance
by about 11.4–11.5 minutes in the measurement window. Actual waveform changes
then contribute roughly +14 minutes, reversing the centroid sign. Thus the
window correction alone does not explain these delays.

Both edits replace two parallel parent links with one child link between the
same endpoints: example 02 removes 58/59 and adds 79 (40→39); example 22 removes
13/14 and adds 993 (9→10). No shared link changes endpoints. These two cases
have no evidence of an added topological branch. Derived depth rises and
travel time falls. The saved Cunge diffusion variance drops from about 723 to
566 h² in 02, and 1,261 to 964 h² in 22. Path-mixing variance also falls.
This supports investigating reduced spreading and waveform redistribution;
isolating that mechanism requires a controlled routing experiment.

![Simple delay hydrographs and differences](simple_delays.png)

## Prediction successes and misses

The depth-informed held-network model identifies 55/61 island delays and
51/67 compound-member delays above one minute. Of the remaining cases,
one island and five members are predicted to advance by >1 minute;
five islands and eleven members are predicted within ±1 minute.
Both simple delays are predicted to advance. Correct delay classification
does not imply accurate magnitude. Complete classification tables for both
the geometry and depth-informed models are in classification_counts.csv.

Selected cases below use the depth-informed model. Successful examples have
the smallest absolute error among detected delays >5 minutes; misses have
the largest actual delay; false alarms have the largest predicted delay
among actual advances. Controls are advancing cases of the same type and
network nearest in log parent length and width. Distances and selection
roles are recorded in selected_cases.csv. These are descriptive comparators,
not independently randomized or statistically independent matched samples.

'''+table(result[['example','state','observed_min','predicted_min','transport_min','window_correction_min','shape_residual_min']])+'''

The island 22_101 success is predominantly a transport delay with a window
correction. Island 15_023 combines slower transport and a positive shape
term; the model misses its sign. The 41_041 island false alarm and 37/unit265
member false alarm show the opposite problem: mean transport becomes slower,
but the shape contribution reverses the finite-window centroid response.
A transport descriptor alone cannot reproduce all these signs.

## K, X, and topology

In the saved implementation K = cell length / celerity (seconds), and
X = 0.5 × (1 − hydraulic length scale / cell length). For a reach with N
cells, its transfer mean is N K and its diffusion variance is N K²(1−2X).
Network branch probabilities determine mean travel time and additional
path-mixing variance. K must therefore be interpreted with cell count and
flow fractions; comparing unweighted mean K or X is inadequate. X may be
negative in this implementation; negative X alone does not establish a bug.

mechanisms.csv records full-network splits, joins, cycle rank (with parallel
links retained), cell counts, K/X ranges, transfer moments, and link-ID changes.
routing_links.parquet contains parent/child link-level hydraulics and flow
fractions. Added IDs are replacement candidates, not proof of anomalous branches.
Local lineage/geometry review is still needed for island and compound cases.

## Next tests

1. For the two simple cases, isolate transport versus spreading with compatible
   K/X controls and check time-step convergence, preserving the frozen baseline.
   Evaluate centroid, peak and volume over the shared window and extended support.
2. Inspect island/member edit geometry and lineage, including the large shape
   reversals. Compute local split/rejoin changes, path-time dispersion, branch
   imbalance, nesting and connector changes. Distinguish replacement links
   from genuinely introduced alternate routes.
3. Join island polygon area, elongation, orientation and branch asymmetry.
   No new island polygon descriptors have been measured in this first pass.
4. Relate those measurements to both frozen descriptor sets and their held-out
   errors across the full delay population, retaining near-zero cases separately
   and respecting network/compound-family dependence.

## Verification and reproduction

Run RAPID/investigations/cap_investigation_20261020/delay_side_project.py with
the river-hierarchy-rivgraph environment. Source hashes are in manifest.json.
All 14 selected unique cases reproduce saved centroid shifts within 1e-6 min;
transfer means independently agree with flow-weighted length/celerity.
Translation uses available saved coverage without padding in every case.
Sources are rehashed after analysis. No model fits, routing reruns, or changes
to existing analysis products were made.
'''
    (DEST/'REPORT.md').write_text(report)
    for path,digest in SOURCES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (DEST/'manifest.json').write_text(json.dumps(dict(sources=SOURCES, fits=0, reroutes=0,
       selection='Two simple delays; per type: smallest absolute error among >5min detected delays, largest observed missed delay, largest predicted false delay; nearest same-network/type advancing control by unscaled Euclidean log(parent length,width).',
       limits='Descriptive selection, not independent validation. No polygon shape measurements yet. Added link IDs alone do not establish anomalous branches. Shape residual is not uniquely diffusion.',
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(counts.to_string())
    print(result[['example','state','observed_min','predicted_min','transport_min','window_correction_min','shape_residual_min','translation_padded']].to_string(index=False))


if __name__ == '__main__':
    main()
