"""Controlled routing comparison: frozen saved allocation/depth/n, accepted geometry/slopes.

Synthetic smooth pulse; common spatial discretisation and two clocks per pair.
This does not establish a production reference-allocation or Manning-depth policy.
"""
from pathlib import Path
import sys, json, hashlib, math, time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "RAPID/src")]
import numpy as np
import pandas as pd
from rapid_tools.cunge import topology
from rapid_tools.common_grid import route_interval_means
from rapid_tools.k_values import compute_dt_bounds

OUT = ROOT / "RAPID/outputs/local_head_routing_pilot_v2_20261028"
STAGE = ROOT / "RAPID/outputs/local_head_integration_verified_20261028"
STATES = ["S013_unit_8", "S296_unit_30", "S023_unit_136", "S244_unit_262"]


def parameters(source, counts):
    f = source.copy()
    f["count"] = counts
    f["rapid_effective_length_m"] = f.link_length_m / f["count"]
    f["celerity_times_n"] = 5 / 3 * f.depth_m ** (2 / 3) * np.sqrt(f.slope)
    f["rapid_celerity_mps_raw"] = f.celerity_times_n / f.n_manning
    f["rapid_celerity_mps"] = np.clip(
        f.rapid_celerity_mps_raw, f.celerity_min_mps, f.celerity_max_mps
    )
    f["rapid_celerity_capped"] = ~np.isclose(
        f.rapid_celerity_mps, f.rapid_celerity_mps_raw, rtol=1e-12, atol=1e-12
    )
    f["rapid_k"] = f.rapid_effective_length_m / f.rapid_celerity_mps
    f["hydraulic_length_scale_m"] = f.reference_discharge_cms / (
        f.width * f.slope * f.rapid_celerity_mps
    )
    f["rapid_x"] = (
        0.5
        - f.diffusion_multiplier
        * 0.5
        * f.hydraulic_length_scale_m
        / f.rapid_effective_length_m
    )
    f["hydraulic_diffusivity_m2_s"] = f.reference_discharge_cms / (
        2 * f.width * f.slope
    )
    if "kb_local" in f:
        f = f.drop(columns="kb_local")
    return f


def moments(f):
    order, edges, root, end = topology(f)
    mass = {root: 1.0}
    one = {root: 0.0}
    two = {root: 0.0}
    for u in order:
        for i in edges.get(u, []):
            r = f.iloc[i]
            w = mass[u] * r.split_fraction
            mu = one[u] / mass[u]
            m2 = two[u] / mass[u]
            delay = r["count"] * r.rapid_k
            var = r["count"] * 2 * r.rapid_k**2 * (1 - 2 * r.rapid_x)
            mass[r.v] = mass.get(r.v, 0.0) + w
            one[r.v] = one.get(r.v, 0.0) + w * (mu + delay)
            two[r.v] = two.get(r.v, 0.0) + w * (m2 + 2 * mu * delay + delay**2 + var)
    return one[end], max(0.0, two[end] - one[end] ** 2) ** 0.5


def main():
    OUT.mkdir(exist_ok=False)
    sources = {}
    rows = []
    curves = {}
    hydraulics = []

    def track(p):
        p = Path(p)
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return p

    plan = json.loads(
        track(
            ROOT / "RAPID/outputs/uncapped_event_window_20261020/routing_plan.json"
        ).read_text()
    )
    case = next(c for c in plan["cases"] if c["example"] == "22")
    jobs = {j["state_id"]: j for j in case["jobs"]}
    for state in STATES:
        source = pd.read_parquet(track(jobs[state]["parameters"])).reset_index(
            drop=True
        )
        candidate = pd.read_parquet(
            track(STAGE / "22" / state / "slope_candidate.parquet")
        ).set_index("id_link")
        if set(source.id_link) != set(candidate.index):
            raise ValueError("Width/depth lineage review required for restored IDs")
        candidate = candidate.loc[source.id_link].reset_index()
        np.testing.assert_array_equal(source[["u", "v"]], candidate[["u", "v"]])
        corrected = source.copy()
        for col in ["link_length_m", "head_us", "head_ds", "slope"]:
            corrected[col] = candidate[col].to_numpy()
        for col in [
            "width",
            "depth_m",
            "n_manning",
            "split_fraction",
            "reference_flow_fraction",
            "reference_discharge_cms",
            "diffusion_multiplier",
        ]:
            np.testing.assert_array_equal(source[col], corrected[col])
        # Reproduce coefficients before choosing a shared accuracy grid.
        check = parameters(source, source["count"])
        np.testing.assert_allclose(
            check[["rapid_k", "rapid_x"]],
            source[["rapid_k", "rapid_x"]],
            rtol=1e-12,
            atol=1e-12,
        )
        required = []
        for f in (source, corrected):
            t = parameters(f, f["count"])
            required.append(
                np.maximum(
                    1,
                    np.ceil(
                        t.link_length_m / (t.cunge_alpha * t.hydraulic_length_scale_m)
                    ),
                ).astype(int)
            )
        counts = np.maximum.reduce([source["count"].to_numpy(), *required])
        frames = {
            "existing": parameters(source, counts),
            "corrected": parameters(corrected, counts),
        }
        upper = min(
            compute_dt_bounds(
                f.rapid_k, x_value=f.rapid_x, coefficient_policy="signed_inflow"
            )[1]
            for f in frames.values()
        )
        interval = 300.0
        nsub = math.ceil(interval / min(15.0, 0.99 * upper))
        dt = interval / nsub
        predicted = [moments(f) for f in frames.values()]
        duration = max(mu + 40 * sd + 72 * 3600 for mu, sd in predicted)
        t = (np.arange(math.ceil(duration / interval)) + 0.5) * interval
        pulse = np.where(t < 24 * 3600, 50 * np.sin(np.pi * t / (24 * 3600)) ** 2, 0.0)
        flow = 100 + pulse
        mass_in = float(pulse.sum() * interval)
        pd.DataFrame(dict(seconds=t, forcing_cms=flow)).to_parquet(
            OUT / f"{state}_forcing.parquet", index=False
        )
        for option, f in frames.items():
            folder = OUT / state / option
            folder.mkdir(parents=True)
            f.to_parquet(folder / "parameters.parquet", index=False)
            for clock in (1, 2):
                started = time.monotonic()
                q, audit = route_interval_means(
                    f, flow, nsub * clock, dt / clock, initial_flow=100.0
                )
                excess = q - 100.0
                volume = float(excess.sum() * interval)
                centroid = float(t @ excess / excess.sum())
                if audit["minimum_native_flow_cms"] < -1e-9:
                    raise ValueError("Negative routed discharge")
                mass_error = abs(volume / mass_in - 1)
                tail = abs(q[-1] - 100.0)
                if mass_error > 1e-5 or tail > 1e-4:
                    raise ValueError(
                        f"Extend pulse tail: volume error={mass_error}, tail={tail}"
                    )
                rows.append(
                    dict(
                        state=state,
                        option=option,
                        clock=clock,
                        dt_s=dt / clock,
                        peak_cms=float(q.max()),
                        centroid_s=centroid,
                        volume_relative_error=mass_error,
                        tail_error_cms=tail,
                        seconds=time.monotonic() - started,
                        **audit,
                    )
                )
                pd.DataFrame(dict(seconds=t, outlet_cms=q)).to_parquet(
                    folder / f"outlet_clock{clock}.parquet", index=False
                )
                curves[state, option, clock] = q
                print(state, option, clock, "mass error", mass_error, flush=True)
        hydraulic = dict(
            state=state,
            shared_grid_refined_links=int(np.sum(counts != source["count"])),
            length_changes=int(
                np.sum(abs(corrected.link_length_m - source.link_length_m) > 1e-6)
            ),
            slope_changes=int(np.sum(abs(corrected.slope / source.slope - 1) > 1e-8)),
        )
        for col in ("rapid_k", "rapid_x"):
            a, b = frames["existing"][col], frames["corrected"][col]
            hydraulic[col + "_changed_links"] = int(
                np.sum(~np.isclose(a, b, rtol=1e-10, atol=1e-12))
            )
        hydraulics.append(hydraulic)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "routing_audit.csv", index=False)
    comparisons = []
    for state in STATES:
        a = table[(table.state == state) & (table.option == "existing")].set_index(
            "clock"
        )
        b = table[(table.state == state) & (table.option == "corrected")].set_index(
            "clock"
        )
        comparisons.append(
            dict(
                state=state,
                peak_change_cms=float(b.loc[2, "peak_cms"] - a.loc[2, "peak_cms"]),
                centroid_change_s=float(
                    b.loc[2, "centroid_s"] - a.loc[2, "centroid_s"]
                ),
                max_outlet_change_cms=float(
                    np.max(
                        abs(
                            curves[state, "corrected", 2] - curves[state, "existing", 2]
                        )
                    )
                ),
                max_clock_difference_cms=max(
                    float(np.max(abs(curves[state, o, 2] - curves[state, o, 1])))
                    for o in frames
                ),
                centroid_effect_clock_error_s=float(
                    abs(
                        (b.loc[2, "centroid_s"] - a.loc[2, "centroid_s"])
                        - (b.loc[1, "centroid_s"] - a.loc[1, "centroid_s"])
                    )
                ),
            )
        )
    pd.DataFrame(comparisons).to_csv(OUT / "response_comparison.csv", index=False)
    pd.DataFrame(hydraulics).to_csv(OUT / "hydraulic_comparison.csv", index=False)
    for p, h in sources.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise ValueError("Source modified during routing")
    track(__file__)
    (OUT / "manifest.json").write_text(
        json.dumps(
            dict(
                source_hashes=sources,
                production_modified=False,
                rf_training_runs=0,
                routing_runs=len(rows),
                depth_policy="saved BASED fixed",
                allocation_policy="saved per-case allocation fixed; not original-reference allocation correction",
                forcing="24h sine-squared pulse, baseline 100 cms, amplitude 50 cms",
                grid_policy="common counts per existing/corrected pair",
                comparisons=comparisons,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
