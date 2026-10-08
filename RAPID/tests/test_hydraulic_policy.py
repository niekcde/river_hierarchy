from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from rapid_tools.k_values import KValueConfig, compute_k_values, compute_dt_bounds, compute_routing_dt_seconds
from rapid_tools.hydraulic_policy import production_prep_config
from rapid_tools.prep import prepare_state
from hierarchy_level_definition.metrics.bubble_equivalent_metrics import BubbleEquivalentResistanceConfig, prepare_bubble_resistance_edges


def test_routing_resistance_share_anchored_depth_and_override_stale_kb():
    reference = dict(kb_value=50., kb_source_method="stored_based", reference_width_m=500.,
                     reference_depth_m=10., reference_slope=1e-4, reference_discharge_cms=1000.)
    links = pd.DataFrame(dict(id_link=[1,2,3], id_us_node=[0,1,2], id_ds_node=[1,2,3],
                              wid_adj_routing=[5.,500.,1000.], link_length_m=[200.]*3,
                              slope_used=[1e-4]*3, k_b=[99.]*3))
    routed = compute_k_values(links, config=KValueConfig(depth_width_exponent=.65,
        kb_reference_width_m=500., kb_reference_depth_m=10., n_manning=.035))
    resistance, _ = prepare_bubble_resistance_edges(links, config=BubbleEquivalentResistanceConfig(
        kb_mode="based_reference_section", reference_kb_summary=reference,
        depth_width_exponent=.65, resistance_law="chezy"))
    expected = 10.*(links.wid_adj_routing/500.)**.65
    np.testing.assert_allclose(routed.rapid_depth_m, expected)
    np.testing.assert_allclose(resistance.d, expected)
    np.testing.assert_allclose(resistance.k_b, routed.rapid_kb_value)


def test_signed_policy_removes_only_lower_clock_bound():
    k = pd.Series([40., 4000.])
    assert compute_dt_bounds(k, x_value=.1) == (800.,72.)
    assert compute_dt_bounds(k, x_value=.1, coefficient_policy="signed_current_inflow") == (0.,72.)
    assert compute_routing_dt_seconds(k, x_value=.1, forcing_dt_seconds=3600,
        coefficient_policy="signed_current_inflow", preferred_routing_dt_seconds=72) == 72
    with pytest.raises(ValueError, match="No stable"):
        compute_routing_dt_seconds(k, x_value=.1, forcing_dt_seconds=3600)


def test_signed_policy_rejects_timestep_dependent_hydraulics():
    from dataclasses import replace
    for name in ("use_stability_effective_length_cap", "use_routing_celerity_regularization"):
        with pytest.raises(ValueError, match="keep hydraulic K fixed"):
            prepare_state(None, prep_config=replace(production_prep_config(), **{name: True}))


def test_new_depth_policy_cannot_import_old_transition_results(tmp_path, monkeypatch):
    import network_variants.equivalent_resistance_experiment as experiment
    pd.DataFrame([dict(state_id="S000_base", state_role="base", status="complete",
                       mode="independent-units")]).to_csv(tmp_path/"state_registry.csv", index=False)
    (tmp_path/"equivalent_transition_registry.csv").write_text("transition_id\nT001\n")
    monkeypatch.setattr(experiment, "load_manifest_equivalent_resistance_config",
                        lambda path: (BubbleEquivalentResistanceConfig(), {}))
    with pytest.raises(ValueError, match="different depth-width policy"):
        experiment.run_equivalent_resistance_experiment(tmp_path,
            equivalent_resistance_config=BubbleEquivalentResistanceConfig(depth_width_exponent=.65))


def test_production_engine_starts_at_the_forcing_plateau(tmp_path):
    from dataclasses import replace
    from test_rapid_prep import _write_state
    from rapid_tools.registry import iter_preparable_states, load_state_registry
    from rapid_tools.engine import run_prepared_state
    experiment = _write_state(tmp_path)
    context = next(iter_preparable_states(load_state_registry(experiment)))
    config = replace(production_prep_config(), kb_mode="fixed", depth_width_exponent=None)
    forcing = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=10, freq="60s", tz="UTC"),
                            "discharge_cms": [7.]*10})
    prepare_state(context, prep_config=config, forcing_table=forcing, planned_routing_dt_seconds=10)
    _, qout = run_prepared_state(context.rapid_prep_dir, return_qout=True)
    np.testing.assert_allclose(qout, 7., atol=1e-12)
    with pytest.raises(ValueError, match="upper bound"):
        run_prepared_state(context.rapid_prep_dir, routing_timestep_seconds=10**8)
    from rapid_tools.engine import _publish_compact_result, RapidResultStorageConfig, _remove_disposable_prep
    from rapid_tools.analysis import load_reach_wave_stack
    context.rapid_run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_prepared_state(context.rapid_prep_dir, output_path=context.rapid_run_dir/"Qout.nc")
    original = pd.read_csv(context.rapid_prep_dir/"rapid_link_attributes.csv")
    published = _publish_compact_result(prep_dir=context.rapid_prep_dir, run_dir=context.rapid_run_dir,
        qout_path=raw, storage=RapidResultStorageConfig(compact_qout=True, compact_geometry=True,
            write_mapping_parquet=True, retain_prepared_inputs=False))
    _remove_disposable_prep(context.rapid_prep_dir)
    wave = load_reach_wave_stack(context.rapid_prep_dir, published["qout_nc"], run_dir=context.rapid_run_dir)
    np.testing.assert_allclose(wave.qout, 7., atol=1e-12)
    assert set(original.columns).issubset(wave.reaches.columns)
    expected = original.set_index("reach_id").rapid_k.reindex(wave.reach_ids)
    np.testing.assert_allclose(wave.reaches.rapid_k, expected)


def test_fractional_clock_preserved_by_production_preparation_and_engine(tmp_path):
    from dataclasses import replace
    import json
    from test_rapid_prep import _write_state
    from rapid_tools.registry import iter_preparable_states, load_state_registry
    from rapid_tools.engine import run_prepared_state, stp_cor
    experiment = _write_state(tmp_path)
    context = next(iter_preparable_states(load_state_registry(experiment)))
    config = replace(production_prep_config(), kb_mode="fixed", depth_width_exponent=None)
    forcing = pd.DataFrame({"time": pd.date_range("2020-01-01", periods=10, freq="1085s", tz="UTC"),
                            "discharge_cms": [7.,7.,8.,10.,12.,10.,8.,7.,7.,7.]})
    dt = 1085/16
    prepare_state(context, prep_config=config, forcing_table=forcing, planned_routing_dt_seconds=dt)
    manifest = json.loads((context.rapid_prep_dir/'rapid_prep_manifest.json').read_text())
    assert manifest['routing']['routing_dt_seconds'] == dt
    qpath, qout = run_prepared_state(context.rapid_prep_dir, return_qout=True)
    assert json.loads(qpath.with_suffix('.audit.json').read_text())['routing_dt_seconds'] == dt
    assert np.isfinite(qout).all() and (qout >= 0).all()
    assert stp_cor(1085,dt)==16
    with pytest.raises(ValueError):stp_cor(1085,67.)


def test_explicit_fractional_clock_bounds_and_divisibility():
    from rapid_tools.k_values import compute_routing_dt_seconds_from_bounds
    assert compute_routing_dt_seconds_from_bounds(dt_min=0,dt_max=96,forcing_dt_seconds=755,
        preferred_routing_dt_seconds=94.375)==94.375
    for dt in [100.5,94.4,float('nan')]:
        with pytest.raises(ValueError):
            compute_routing_dt_seconds_from_bounds(dt_min=0,dt_max=96,forcing_dt_seconds=755,
                preferred_routing_dt_seconds=dt)
