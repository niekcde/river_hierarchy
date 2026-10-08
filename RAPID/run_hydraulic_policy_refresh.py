"""Prepare audited states using production APIs, with fresh versioned outputs.

The original experiment trees are read-only. Identical hydraulic states share
work; inventory aliases retain all scientific comparisons. --route uses saved
event forcing only, never silently substitutes an artificial event.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import hashlib
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RAPID/src")]
import numpy as np
import pandas as pd
from rapid_tools.hydraulic_policy import POLICY_VERSION, production_prep_config
from rapid_tools.prep import prepare_state
from rapid_tools.registry import RapidStateContext
from rapid_tools.engine import run_prepared_state, _publish_compact_result, _remove_disposable_prep, RapidResultStorageConfig
from rapid_tools.hydrograph import write_hydrograph_outputs
from rapid_tools.forcing import ForcingConfig, prepare_forcing_table, _apply_constant_spinup, infer_forcing_dt_seconds
from rapid_tools.k_values import KValueConfig, compute_k_values
from hierarchy_level_definition.metrics.bubble_equivalent_metrics import BubbleEquivalentResistanceConfig, prepare_bubble_resistance_edges

ASSETS = ROOT / "RAPID/assets/bounded_robust_slopes_20261002"
INVENTORY = ASSETS / "child_head_reconstruction_20261003"
DEFAULT_OUTPUT = ROOT / "RAPID/outputs" / POLICY_VERSION
VALIDATED_FRACTIONAL_CLOCKS = {"37": 94.375, "46": 67.8125}
LEGACY_IMPLEMENTATION = "bc018c7fe205d138bf22e726b0662c8ebde0f4005f3f9c8ab8fa85bf5e15ca0b"


def reference_summary(example):
    r = pd.read_csv(ASSETS / "hydraulic_references.csv", dtype={"example": str}).set_index("example").loc[example]
    return dict(kb_value=r.reference_width_m/r.stored_depth_m,
        kb_source_method="stored_BASED_reference_anchor", reference_width_m=r.reference_width_m,
        reference_depth_m=r.stored_depth_m, reference_discharge_cms=r.reference_q_cms,
        reference_slope=r.new_global_slope)


def run(example, output, *, prepare=False, route=False, states=None, compact=False, resume=False):
    inv = pd.read_csv(INVENTORY/example/"inventory.csv", dtype={"example": str})
    good = inv[inv.status.eq("ready")].drop_duplicates("fingerprint")
    if states:
        selected = inv[inv.state.isin(states)].fingerprint
        good = good[good.fingerprint.isin(selected)]
    dest = output/example
    dest.mkdir(parents=True, exist_ok=True)
    inv["refreshed_state_dir"] = inv.fingerprint.map(lambda value: str(dest/str(value)) if pd.notna(value) else "")
    inv.to_csv(dest/"inventory.csv", index=False)
    ref = reference_summary(example)
    config = production_prep_config()
    method = json.loads((INVENTORY/example/"routing_method.json").read_text())
    dt = method["clocks"][0]
    forcing = None
    if route:
        suffix = "_direction130" if example == "15" else ""
        saved = ROOT/f"network_variants/output/{example}_indep{suffix}__anchored/states/S000_base/rapid/run/outlet_hydrograph.csv"
        if saved.exists():
            h = pd.read_csv(saved)
            forcing = pd.DataFrame(dict(time=pd.to_datetime(h.time_seconds, unit="s", origin="2020-01-01", utc=True), discharge_cms=h.q_inflow_cms))
            forcing_provenance = dict(source="saved_actual_event", path=str(saved))
        else:
            experiment = saved.parents[4]
            recorded = json.loads((experiment/"experiment_manifest.json").read_text())["options"]["hierarchy"]
            event_config = ForcingConfig(**recorded["equivalent_forcing_config"])
            event_config = replace(event_config, output_cache_dir=str(dest/"forcing_cache"))
            forcing, metadata = prepare_forcing_table(recorded["equivalent_forcing_path"], config=event_config)
            # Shared constant padding provides settling time for all children.
            full = inv[inv.status.eq("ready")]
            padding_hours = max(240., 3*full.max_travel_seconds.max()/3600.)
            forcing, _ = _apply_constant_spinup(forcing, spinup_hours=padding_hours, pad_end_with_spinup=True)
            forcing_provenance = dict(source="recorded_gauge_event_window", metadata=metadata,
                event_config=asdict(event_config), shared_padding_hours=padding_hours,
                earlier_investigation_coverage="standard pulse only")
        forcing_dt = infer_forcing_dt_seconds(forcing)
        upper = float(inv[inv.status.eq("ready")].upper_dt_seconds.min())
        dt = max(d for d in range(1, forcing_dt+1) if forcing_dt % d == 0 and d <= .99*upper)
        if example in VALIDATED_FRACTIONAL_CLOCKS:
            dt = VALIDATED_FRACTIONAL_CLOCKS[example]
            quotient = forcing_dt / dt
            if dt > .99*upper or not np.isclose(quotient, round(quotient), rtol=0, atol=1e-9):
                raise ValueError("Validated fractional clock no longer matches the hydraulic inputs or forcing.")
        forcing_provenance["routing_dt_seconds"] = dt
        (dest/"forcing_provenance.json").write_text(json.dumps(forcing_provenance, indent=2, default=str))
    rows = []
    implementation = hashlib.sha256(b"".join(p.read_bytes() for p in [Path(__file__),
        ROOT/"network_variants/reference_context.py", *sorted((ROOT/"RAPID/src/rapid_tools").glob("*.py"))])).hexdigest()
    for r in good.itertuples():
        state_dest = dest/r.fingerprint
        input_bytes = Path(r.frame).read_bytes() + (pd.util.hash_pandas_object(forcing, index=True).values.tobytes() if forcing is not None else b"")
        identity = hashlib.sha256((implementation + json.dumps(asdict(config), sort_keys=True)
            + json.dumps({"routing_dt_seconds": dt}, sort_keys=True)).encode() + input_bytes).hexdigest()
        complete = state_dest/"completed.json"
        if resume and complete.exists():
            previous = json.loads(complete.read_text())
            # Scoped migration of case 22 only: scientific inputs and its
            # integer clock are unchanged. Never reuse old clocks for 37/46.
            legacy_identity = hashlib.sha256((LEGACY_IMPLEMENTATION + json.dumps(asdict(config), sort_keys=True)).encode() + input_bytes).hexdigest()
            compatible_legacy = (example == "22" and previous.get("identity") == legacy_identity
                                 and previous.get("routing_dt_s") == dt)
            if (previous.get("identity") == identity or compatible_legacy) and previous.get("status") == "routed" and Path(previous["qout"]).exists():
                rows.append(previous)
                pd.DataFrame(rows).to_csv(dest/"refresh_status.csv", index=False)
                continue
        f = pd.read_parquet(r.frame)
        local = f.copy()
        local["link_length_m"] = f.rapid_effective_length_m
        local["slope_used"] = f.slope
        actual = compute_k_values(local, config=KValueConfig(
            depth_width_exponent=.65, kb_reference_width_m=ref["reference_width_m"],
            kb_reference_depth_m=ref["reference_depth_m"], n_manning=.035,
            use_celerity_capping=True, min_celerity_mps=.05, max_celerity_mps=5., min_effective_length_m=200.))
        np.testing.assert_allclose(actual.rapid_k, f.rapid_k, rtol=1e-12)
        eqconfig = BubbleEquivalentResistanceConfig(resistance_law="chezy", kb_mode="based_reference_section",
            reference_kb_summary=ref, depth_width_exponent=.65, solver_backend="energy_sparse",
            solver_cache_dir=str(dest/"equivalent_solver_cache"))
        edges, _ = prepare_bubble_resistance_edges(f, config=eqconfig)
        np.testing.assert_allclose(edges.d, f.depth_m, rtol=1e-12)
        state_dest.mkdir(exist_ok=True)
        edges.to_parquet(state_dest/"resistance_edges.parquet", index=False)
        row = dict(example=example, state=r.state, fingerprint=r.fingerprint, status="hydraulics_verified",
                   n_links=len(f), n_segments=int(f["count"].sum()), routing_dt_s=dt)
        if route:
            estimated_raw_bytes = len(forcing)*int(f["count"].sum())*8
            if shutil.disk_usage(dest).free < 2*estimated_raw_bytes + 10*1024**3:
                raise OSError("Insufficient free space for the next state and publication reserve.")
        if prepare or route:
            source = Path(r.candidate_manifest)
            manifest = json.loads(source.read_text())
            contextdir = state_dest/"hydraulic_context"
            contextdir.mkdir(exist_ok=True)
            context = pd.read_csv(source.parent/"link_hydraulic_context.csv")
            context["slope_adjusted"] = True
            context.to_csv(contextdir/"link_hydraulic_context.csv", index=False)
            shutil.copy2(source.parent/"node_hydraulic_context.csv", contextdir/"node_hydraulic_context.csv")
            manifest["output_paths"] = {"link_hydraulic_context": str(contextdir/"link_hydraulic_context.csv"),
                                        "node_hydraulic_context": str(contextdir/"node_hydraulic_context.csv")}
            manifest["policy_version"] = POLICY_VERSION
            (contextdir/"hydraulic_context_manifest.json").write_text(json.dumps(manifest, indent=2))
            paths = manifest["input_paths"]
            c = RapidStateContext(r.state, state_dest, Path(paths["directed_links"]), Path(paths["directed_nodes"]),
                Path(paths["directed_links"]), None, contextdir, state_dest/"prep", state_dest/"run",
                "complete", 0, "base" if r.state == "S000_base" else "derived")
            prepare_state(c, prep_config=config, reference_kb_summary=ref, forcing_table=forcing,
                planned_routing_dt_seconds=dt if route else None)
            attrs = pd.read_csv(c.rapid_prep_dir/"rapid_link_attributes.csv", low_memory=False)
            # Each original link must have the same audited K on all its segments.
            key = "parent_link_id" if "parent_link_id" in attrs else "id_link"
            np.testing.assert_array_equal(attrs.groupby(key).size().reindex(f.id_link).to_numpy(), f["count"].to_numpy())
            expected = attrs[key].map(f.set_index("id_link").rapid_k)
            np.testing.assert_allclose(attrs.rapid_k, expected, rtol=1e-10)
            row["status"] = "prepared_verified"
            if route:
                qpath = run_prepared_state(c.rapid_prep_dir, output_path=c.rapid_run_dir/"Qout.nc")
                audit = json.loads(qpath.with_suffix(".audit.json").read_text())
                if audit["minimum_substep_flow_cms"] < -1e-10*max(1., audit["maximum_substep_flow_cms"]):
                    raise ValueError(f"Negative routed flow beyond floating-point tolerance: {audit}")
                row.update(status="routed", qout=str(qpath))
                write_hydrograph_outputs(c.rapid_prep_dir, qpath, c.rapid_run_dir)
                if compact:
                    published = _publish_compact_result(prep_dir=c.rapid_prep_dir, run_dir=c.rapid_run_dir,
                        qout_path=qpath, storage=RapidResultStorageConfig(compact_qout=True,
                            retain_prepared_inputs=False, write_mapping_parquet=True, compact_geometry=True))
                    row["qout"] = published["qout_nc"]
                    _remove_disposable_prep(c.rapid_prep_dir)
        (state_dest/"policy_manifest.json").write_text(json.dumps(dict(policy_version=POLICY_VERSION,
            source_candidate_manifest=r.candidate_manifest, reference_kb_summary=ref,
            prep_config=asdict(config), equivalent_resistance_config=asdict(eqconfig)), indent=2))
        row["identity"] = identity
        complete.write_text(json.dumps(row, indent=2))
        rows.append(row)
        pd.DataFrame(rows).to_csv(dest/"refresh_status.csv", index=False)
        print(example, r.state, row["status"], flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("examples", nargs="*")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--route", action="store_true")
    p.add_argument("--states", nargs="+")
    p.add_argument("--compact", action="store_true", help="Compress full Qout and retain mapping; remove disposable prep after publication.")
    p.add_argument("--resume", action="store_true", help="Reuse only completed runs with matching code, hydraulic frame, policy and forcing.")
    args = p.parse_args()
    examples = args.examples or sorted(pd.read_csv(INVENTORY/"inventory.csv", dtype={"example": str}).example.unique())
    for ex in examples:
        run(ex, args.output, prepare=args.prepare, route=args.route, states=args.states, compact=args.compact, resume=args.resume)


if __name__ == "__main__":
    main()
