"""Export the selected reviewed graph to GPKG, then prepare, route and predict it.

Uses the existing RAPID Cunge implementation and frozen calibration. This
materializes the audited graph composition, not a new combined water raster.
"""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'RAPID/src'),
               str(ROOT / 'RAPID/investigations/hydrograph_predictors_20261014')]
import geopandas as gpd
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from rapid_tools import cunge, common_grid
from rapid_tools.hydrograph_response import load, distribution, digest
from build import descriptors, pair_features
import final_forest_search as forest

SOURCE = ROOT / 'RAPID/outputs/example48_combined_selection_20261019/domain_budget_10'
OUT = ROOT / 'RAPID/outputs/example48_selected_network_20261019'
BASE = ROOT / 'network_variants/output/48_indep__anchored/states/S000_base'
SELECTED = [2, 12, 23, 27, 28, 34, 39, 44, 48, 49, 54, 59, 61]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def node_layer(links, base_nodes):
    graph = nx.MultiDiGraph()
    graph.add_edges_from(zip(links.u, links.v))
    rows = []
    for node in sorted(graph):
        incident = links[(links.u == node) | (links.v == node)]
        endpoints = [(Point(r.geometry.coords[0]), Point(r.geometry.coords[-1]))
                     for r in incident.itertuples()]
        candidates = [p for pair in endpoints for p in pair]
        if node in base_nodes.index:
            candidates.insert(0, base_nodes.loc[node].geometry)
        point = min(candidates, key=lambda p: sum(min(p.distance(a), p.distance(b)) for a, b in endpoints))
        gap = max(min(point.distance(a), point.distance(b)) for a, b in endpoints)
        assert gap < 1e-5, (node, gap)
        rows.append(dict(id_node=int(node), in_degree=graph.in_degree(node),
                         out_degree=graph.out_degree(node), is_inlet=graph.in_degree(node) == 0,
                         is_outlet=graph.out_degree(node) == 0, endpoint_gap_m=gap, geometry=point))
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=links.crs)


def main():
    OUT.mkdir(exist_ok=True)
    prior = json.loads((SOURCE / 'result.json').read_text())
    assert prior['units'] == SELECTED
    params = pd.read_parquet(SOURCE / 'parameters.parquet')
    geom = gpd.read_parquet(SOURCE / 'links.parquet')
    np.testing.assert_array_equal(params[['id_link', 'u', 'v']], geom[['id_link', 'u', 'v']])
    columns = ['id_link', 'id_us_node', 'id_ds_node', 'u', 'v', 'link_length_m',
               'width', 'depth_m', 'slope', 'head_us', 'head_ds', 'kb_local',
               'n_manning', 'diffusion_multiplier', 'cunge_alpha', 'grid_n_min',
               'grid_n_max', 'celerity_min_mps', 'celerity_max_mps']
    links = gpd.GeoDataFrame(params[columns].copy(), geometry=geom.geometry, crs=geom.crs)
    np.testing.assert_allclose(links.length, links.link_length_m, rtol=0, atol=1e-7)
    base = gpd.read_file(BASE / 'variant/directed/48__S000_base_directed_links.gpkg')
    base_nodes = gpd.read_file(BASE / 'variant/directed/48__S000_base_directed_nodes.gpkg').set_index('id_node')
    nodes = node_layer(links, base_nodes)
    points = nodes.set_index('id_node').geometry
    # Deliver line coordinates in the same upstream-to-downstream order as u/v.
    links.geometry = [line if Point(line.coords[0]).distance(points.loc[u]) < 1e-5
                      else LineString(list(line.coords)[::-1]) for line, u in zip(links.geometry, links.u)]
    for row in links.itertuples():
        assert Point(row.geometry.coords[0]).distance(points.loc[row.u]) < 1e-5
        assert Point(row.geometry.coords[-1]).distance(points.loc[row.v]) < 1e-5
    links['example'] = '48'
    links['state'] = 'selected_intermediate'
    package = OUT / 'example48_selected_network.gpkg'
    links.to_file(package, layer='intermediate_links', driver='GPKG', index=False)
    nodes.to_file(package, layer='intermediate_nodes', driver='GPKG', index=False)
    giscols = ['id_link', 'id_us_node', 'id_ds_node', 'geometry']
    base[giscols].to_file(package, layer='base_links', driver='GPKG', index=False)
    center = gpd.read_file(ROOT / 'network_variants/output/48_indep__anchored/reference_section/variant/directed/48__reference_section_directed_links.gpkg')
    center[giscols].to_file(package, layer='single_centerline_links', driver='GPKG', index=False)
    membership = pd.read_csv(BASE / 'variant/summary/unit_link_membership.csv')
    selected = membership[membership.unit_id.isin(SELECTED)]
    selection = base[giscols].merge(selected[['unit_id', 'id_link']], on='id_link')
    selection.to_file(package, layer='selected_base_unit_links', driver='GPKG', index=False)
    pd.DataFrame({'selected_unit_id': SELECTED}).to_csv(OUT / 'selected_units.csv', index=False)

    # Prepare exclusively from the delivered GIS attributes, not cached K/X.
    exported = gpd.read_file(package, layer='intermediate_links').sort_values('id_link')
    physical = pd.DataFrame(exported.drop(columns='geometry'))
    graph = nx.MultiDiGraph()
    graph.add_edges_from(zip(physical.u, physical.v))
    assert nx.is_directed_acyclic_graph(graph) and nx.is_weakly_connected(graph)
    assert graph.number_of_edges() - graph.number_of_nodes() + 1 == 49
    config = params.iloc[0]
    reference_q = float(params.loc[params.reference_flow_fraction > 1 - 1e-12, 'reference_discharge_cms'].iloc[0])
    grid = cunge.freeze_grid(physical, reference_q,
                            n_bounds=(config.grid_n_min, config.grid_n_max), alpha=config.cunge_alpha,
                            min_celerity=config.celerity_min_mps, max_celerity=config.celerity_max_mps)
    prepared = cunge.parameters(grid, config.n_manning, beta=config.diffusion_multiplier)
    for col in ['width', 'depth_m', 'slope', 'rapid_k', 'rapid_x', 'count', 'reference_flow_fraction']:
        np.testing.assert_allclose(prepared[col], params.sort_values('id_link')[col], rtol=1e-12, atol=1e-12)
    states, responses = load('0.5%')
    states = states[states.example == '48'].set_index('state')
    plan = json.loads((ROOT / 'RAPID/outputs/event_window_plan_20261013/plan.json').read_text())
    case = next(c for c in plan['cases'] if c['example'] == '48')
    forcing_path = ROOT / 'RAPID/outputs/event_window_runs_20261013/48/tail_120h/forcing.parquet'
    forcing = pd.read_parquet(forcing_path)
    prep = OUT / 'rapid/cunge/prep'
    run = OUT / 'rapid/cunge/run'
    run.mkdir(parents=True, exist_ok=True)
    prep_manifest = cunge.write_prepared_state(prepared, forcing.forcing_cms.to_numpy(), 60, 7.5, prep,
                              output_averaging='linear_state_interval_mean',
                              metadata=dict(example='48', state='selected_intermediate',
                                            geometry=str(package), layer='intermediate_links',
                                            geometry_sha256=digest(package), selected_units=SELECTED))
    prep_manifest['state_role'] = 'selected_intermediate'
    prep_manifest['geometry_scope'] = 'Physical links in example48_selected_network.gpkg/intermediate_links; numerical sections map to those link IDs.'
    save(prep / 'rapid_prep_manifest.json', prep_manifest)
    prepared.to_parquet(prep / 'parameters.parquet', index=False)
    forcing.to_parquet(prep / 'forcing.parquet', index=False)
    # Native routing with an explicit equilibrium initial condition, reporting
    # interval means. This is the same engine as the finalized response dataset.
    q, audit = common_grid.route_interval_means(prepared, forcing.forcing_cms.to_numpy(),
                                                8, 7.5, initial_flow=case['initial_cms'])
    assert np.isfinite(q).all() and audit['minimum_native_flow_cms'] >= -1e-9
    assert float(np.max(abs(q[-96*60:] - case['final_cms']))) < case['tolerance_cms']
    old = pd.read_parquet(SOURCE / 'outlet.parquet')
    np.testing.assert_allclose(q, old.q_outlet_cms, rtol=0, atol=1e-7)
    outlet = forcing[['interval_start_utc', 'interval_end_utc', 'time_utc', 'phase']].copy()
    outlet['q_outlet_cms'] = q
    outlet.to_parquet(run / 'outlet.parquet', index=False)
    outlet.to_csv(run / 'outlet.csv', index=False)
    parent = states.loc['S000_base']
    t = forcing.time_utc.astype('int64').to_numpy() / 1e9
    a, b = parent.rise_seconds, parent.fall_seconds
    tt = np.r_[a, t[(t > a) & (t < b)], b]
    excess = np.interp(tt, t, q) - parent.baseline_cms
    dist = distribution(tt, excess)
    result = dict(example='48', selected_units=SELECTED, links=len(exported), nodes=len(nodes),
                  base_cycles=63, intermediate_cycles=49, cycles_removed=14,
                  centroid_shift_minutes=(dist['centroid_seconds'] - parent.centroid_seconds) / 60,
                  peak_shift_minutes=(tt[excess.argmax()] - parent.peak_seconds) / 60,
                  peak_excess_change_pct=100 * (excess.max() / parent.peak_excess_cms - 1),
                  duration80_change_pct=100 * (dist['duration80_hours'] / parent.duration80_hours - 1),
                  volume_change_pct=100 * (dist['volume_m3'] / parent.volume_m3 - 1),
                  native_dt_seconds=7.5, report_dt_seconds=60, window='0.5%',
                  window_start=parent.rise_utc, window_end=parent.fall_utc,
                  routing_implementation='rapid_tools.common_grid.route_interval_means / cunge.route',
                  previous_run_max_difference_cms=float(abs(q - old.q_outlet_cms.to_numpy()).max()), audit=audit)

    # Recompute frozen predictors using the exported geometry and attributes.
    artifact = joblib.load(forest.OUT / 'final_model.joblib')
    domains = pd.read_csv(SOURCE / 'domain_predictions.csv')
    directed = nx.DiGraph(graph)
    prediction_rows = []
    for domain in domains.itertuples():
        s, e = int(domain.parent_source), int(domain.parent_sink)
        within = (nx.descendants(directed, s) | {s}) & (nx.ancestors(directed, e) | {e})
        child = prepared[prepared.u.isin(within) & prepared.v.isin(within)]
        parent_graph = pd.read_parquet(ROOT / f'RAPID/outputs/hydrograph_predictor_scopes_20261014/graphs/{domain.parent_domain}.parquet')
        p, c = descriptors(parent_graph), descriptors(child)
        features = pair_features(p, c, 'B_')
        for name in ['route_length_mean', 'route_length_sd', 'width_mean', 'width_cv', 'route_entropy']:
            features['log1p__B_ratio_' + name] = np.log1p(features['B_ratio_' + name])
        features.update(domain_size=p['route_length_mean'] / 1000, B_exposure=1.)
        pred = float(forest.predict(artifact, pd.DataFrame([features]))[0] * 60)
        np.testing.assert_allclose(pred, domain.predicted_minutes, rtol=0, atol=1e-8)
        prediction_rows.append(dict(parent_domain=domain.parent_domain, selected_units=domain.selected_units,
                                    parent_source=s, parent_sink=e, predicted_minutes=pred,
                                    **{k:features[k] for k in artifact['columns']}))
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(OUT / 'predictions_by_domain.csv', index=False)
    result['exploratory_sum_domain_prediction_minutes'] = float(predictions.predicted_minutes.sum())
    result['prediction_note'] = 'Frozen forest per affected domain; sum is exploratory and not a validated whole-network prediction.'
    save(run / 'result.json', result)
    save(OUT / 'results.json', result)
    comparison = pd.DataFrame({'time_utc': forcing.time_utc, 'intermediate_cms': q})
    for state, col in [('S000_base','base_cms'), ('fully_collapsed','single_centerline_cms')]:
        saved = pd.read_parquet(states.loc[state, 'outlet'])
        assert saved.time_utc.equals(forcing.time_utc)
        comparison[col] = saved.q_outlet_cms.to_numpy()
    comparison['in_shared_analysis_window'] = (t >= a) & (t <= b)
    comparison.to_csv(OUT / 'hydrograph_comparison.csv', index=False)
    for name in ['timestep_check.json', 'domain_prediction_validation.csv', 'domain_prediction_summary.json']:
        shutil.copy2(SOURCE / name, OUT / name)
    shutil.copy2(SOURCE.parent / 'domain_budget_10_three_states.png', OUT / 'three_states.png')
    shutil.copy2(SOURCE.parent / 'domain_budget_10_three_states.pdf', OUT / 'three_states.pdf')
    save(OUT / 'manifest.json', dict(selected_units=SELECTED, crs='EPSG:32612',
         geometry_method='Composition of nonoverlapping reviewed anchored replacement graphs; no regenerated combined water raster.',
         source_geometry=str(SOURCE / 'links.parquet'), source_geometry_sha256=digest(SOURCE / 'links.parquet'),
         source_hydraulics_sha256=digest(SOURCE / 'parameters.parquet'),
         model_sha256=digest(forest.OUT / 'final_model.joblib'), forcing_sha256=digest(forcing_path),
         gpkg_sha256=digest(package), script_sha256=digest(__file__),
         validation='GeoPackage reread; endpoints, geometry lengths, connected DAG, cycle rank, hydraulic parameters, full routed outlet and domain predictions verified.',
         result=result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
