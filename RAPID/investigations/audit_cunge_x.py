"""Diagnose constant-reference Cunge X on saved current base-network segments.

Read-only with respect to production: writes separate diagnostic artifacts only.
Run with /opt/anaconda3/envs/geo/bin/python RAPID/investigations/audit_cunge_x.py
"""
from pathlib import Path
import hashlib
import json

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RAPID/validation/current_hydraulic_domain_20261005/provenance.json"
OUT = ROOT / "RAPID/validation/cunge_x_20261005"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flow_shares(frame):
    graph = nx.DiGraph()
    graph.add_edges_from(zip(frame.id_us_node, frame.id_ds_node))
    roots = [v for v in graph if graph.in_degree(v) == 0]
    sinks = [v for v in graph if graph.out_degree(v) == 0]
    assert len(roots) == len(sinks) == 1 and nx.is_directed_acyclic_graph(graph)
    amounts = dict.fromkeys(graph, 0.)
    amounts[roots[0]] = 1.
    shares = {}
    for node in nx.topological_sort(graph):
        edges = frame[frame.id_us_node.eq(node)]
        for edge in edges.itertuples():
            share = amounts[node] * edge.width / edges.width.sum()
            shares[edge.id_link] = share
            amounts[edge.id_ds_node] += share
    assert np.isclose(amounts[sinks[0]], 1.)
    return shares


def summarize(group, example, scenario, x_col):
    x = group[x_col]
    negative = x < 0
    return dict(example=example, scenario=scenario, x_definition=x_col,
                segments=len(group), physical_links=group[['example', 'parent_link_id']].drop_duplicates().shape[0],
                x_min=x.min(), x_p05=x.quantile(.05), x_median=x.median(),
                x_p95=x.quantile(.95), x_max=x.max(),
                negative_segments=int(negative.sum()), negative_percent=100*negative.mean(),
                negative_length_percent=100*group.loc[negative, 'link_length_m'].sum()/group.link_length_m.sum(),
                median_length_for_x_zero_m=group.length_for_x_zero_m.median(),
                median_q_manning_over_allocated=group.q_manning_over_allocated.median())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pieces, refs, sources, coefficients = [], [], [], []
    columns = ['reach_id', 'parent_link_id', 'link_length_m', 'rapid_effective_length_m',
               'slope_used', 'rapid_width_m', 'rapid_depth_m', 'rapid_celerity_mps_raw',
               'rapid_celerity_mps', 'rapid_k', 'rapid_x']
    for source in json.loads(SOURCE.read_text())['sources']:
        example = source['example']
        run = Path(source['hydrograph']).parent
        mapping = run / 'rapid_link_mapping.parquet'
        manifest_path = run.parent / 'policy_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        ref = manifest['reference_kb_summary']
        frame = pd.read_parquet(source['frame'])
        seg = pd.read_parquet(mapping, columns=columns)
        assert seg.reach_id.is_unique
        shares = flow_shares(frame)
        seg['flow_share'] = seg.parent_link_id.map(shares)
        seg['example'] = example
        indexed = frame.set_index('id_link')
        for saved, canonical in [('slope_used','slope'), ('rapid_width_m','width'),
                                 ('rapid_depth_m','depth_m')]:
            np.testing.assert_allclose(seg[saved], seg.parent_link_id.map(indexed[canonical]))
        np.testing.assert_allclose(seg.rapid_k, seg.rapid_effective_length_m/seg.rapid_celerity_mps)
        np.testing.assert_allclose(seg.rapid_x, .1)
        n = manifest['prep_config']['n_manning']
        np.testing.assert_allclose(seg.rapid_celerity_mps_raw,
                                   5/(3*n)*seg.rapid_depth_m**(2/3)*np.sqrt(seg.slope_used))
        hydrograph = pd.read_csv(run/'outlet_hydrograph.csv', usecols=['q_inflow_cms'])
        base, peak = hydrograph.q_inflow_cms.iloc[0], hydrograph.q_inflow_cms.max()
        audit_path = run/'Qout.audit.json'
        dt = json.loads(audit_path.read_text())['routing_dt_seconds']
        qm = .6*seg.rapid_celerity_mps_raw*seg.rapid_width_m*seg.rapid_depth_m
        scenarios = {'stored_reference':ref['reference_discharge_cms'],
                     'event_base':base, 'event_midpoint':(base+peak)/2, 'event_peak':peak,
                     'manning_at_prescribed_depth':None}
        for scenario, total_q in scenarios.items():
            d = seg.copy()
            d['scenario'] = scenario
            d['q_cms'] = qm if total_q is None else total_q*d.flow_share
            d['q_manning_cms'] = qm
            d['q_manning_over_allocated'] = qm/d.q_cms
            d['length_for_x_zero_m'] = d.q_cms/(d.rapid_width_m*d.slope_used*d.rapid_celerity_mps)
            d['x_current_k'] = .5*(1-d.length_for_x_zero_m/d.rapid_effective_length_m)
            d['x_raw_c_physical_length'] = .5*(1-d.q_cms/(d.rapid_width_m*d.slope_used*d.rapid_celerity_mps_raw*d.link_length_m))
            d['x_raw_c_effective_length'] = .5*(1-d.q_cms/(d.rapid_width_m*d.slope_used*d.rapid_celerity_mps_raw*d.rapid_effective_length_m))
            d['implied_diffusion_ratio_current_x'] = .8*d.rapid_effective_length_m/d.length_for_x_zero_m
            # Independent algebraic verification for the coherent Manning state.
            if total_q is None:
                np.testing.assert_allclose(d.x_raw_c_effective_length,
                    .5*(1-3*d.rapid_depth_m/(5*d.slope_used*d.rapid_effective_length_m)), rtol=1e-10, atol=1e-10)
            assert np.isfinite(d.select_dtypes('number')).all().all()
            assert (d.x_current_k < .5).all()
            x, k = d.x_current_k, d.rapid_k
            denominator = dt + 2*k*(1-x)
            current = (dt-2*k*x)/denominator
            previous = (dt+2*k*x)/denominator
            outflow = (2*k*(1-x)-dt)/denominator
            np.testing.assert_allclose(current+previous+outflow, 1.)
            lower = float((2*k*abs(x)).max())
            upper = float((2*k*(1-x)).min())
            coefficients.append(dict(example=example, scenario=scenario, current_dt_seconds=dt,
                current_inflow_coefficient_negative=int((current<0).sum()),
                previous_inflow_coefficient_negative=int((previous<0).sum()),
                previous_outflow_coefficient_negative=int((outflow<0).sum()),
                all_nonnegative_dt_lower_seconds=lower, all_nonnegative_dt_upper_seconds=upper,
                shared_all_nonnegative_interval_exists=lower<=upper))
            pieces.append(d)
        c_ref = 5/(3*n)*ref['reference_depth_m']**(2/3)*np.sqrt(ref['reference_slope'])
        refs.append(dict(example=example, **ref, c_raw_mps=c_ref,
                         x_at_reference_section_200m=.5*(1-ref['reference_discharge_cms']/
                           (ref['reference_width_m']*ref['reference_slope']*c_ref*200)),
                         event_base_cms=base, event_peak_cms=peak,
                         segments=len(seg), physical_links=len(frame),
                         capped_segments=int((~np.isclose(seg.rapid_celerity_mps,seg.rapid_celerity_mps_raw)).sum())))
        sources.append(dict(example=example, mapping=str(mapping), mapping_sha256=sha(mapping),
                            frame=source['frame'], frame_sha256=sha(source['frame']),
                            policy_manifest=str(manifest_path), policy_sha256=sha(manifest_path),
                            audit=str(audit_path), audit_sha256=sha(audit_path),
                            hydrograph=str(run/'outlet_hydrograph.csv'), hydrograph_sha256=sha(run/'outlet_hydrograph.csv')))
    data = pd.concat(pieces, ignore_index=True)
    rows = []
    for example, group in [('all', data), *list(data.groupby('example'))]:
        for scenario, subset in group.groupby('scenario'):
            for col in ['x_current_k', 'x_raw_c_physical_length', 'x_raw_c_effective_length']:
                rows.append(summarize(subset, example, scenario, col))
    summary = pd.DataFrame(rows)
    data.to_parquet(OUT/'segments.parquet', index=False)
    summary.to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame(refs).to_csv(OUT/'reference_sections.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT/'coefficient_diagnostics.csv', index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        scope='16 current real S000_base networks; no simplified children or synthetic networks; no rerouting',
        formula='X=0.5*(1-Q/(B*S*c*dx))',
        main_definition='x_current_k uses saved capped c and effective segment length, preserving current K',
        alternate_definition='x_raw_c_physical_length uses raw c and physical segment length; K would differ where safeguards apply',
        discharge='Steady reference allocation using production width-based bifurcation shares; not transient local flood peaks',
        manning_scenario='Local Manning-implied Q at saved depth; internally consistent per section but not guaranteed network flow conservation',
        reference_section='Single saved reference geometry at illustrative dx=200 m; not a network-wide X',
        negative_x='Retained without clipping. Negative X diagnoses grid/diffusion mismatch with conventional nonnegative X; not by itself proof of instability.',
        checks='Mapping geometry versus canonical frames; K=L/c; raw Manning c; flow allocation closure; algebraic Manning X identity; finite values',
        source_manifest=str(SOURCE), source_manifest_sha256=sha(SOURCE),
        script_sha256=sha(__file__), sources=sources), indent=2))
    print(summary.query("x_definition == 'x_current_k' and (scenario == 'stored_reference' or example == 'all')").to_string(index=False))
    print(pd.DataFrame(refs)[['example','reference_discharge_cms','reference_slope','reference_width_m','x_at_reference_section_200m','capped_segments']].to_string(index=False))


if __name__ == '__main__':
    main()
