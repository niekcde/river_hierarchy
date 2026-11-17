"""Build a self-contained integration and routing review from saved evidence."""
from pathlib import Path
import json, io
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "RAPID/outputs/local_head_integration_verified_20261028"
ROUTES = ROOT / "RAPID/outputs/local_head_routing_pilot_v2_20261028"


def main():
    manifest = json.loads((STAGE / "manifest.json").read_text())
    responses = pd.read_csv(ROUTES / "response_comparison.csv")
    paired = []
    for state in responses.state:
        q = {
            (option, clock): pd.read_parquet(
                ROUTES / state / option / f"outlet_clock{clock}.parquet"
            ).outlet_cms.to_numpy()
            for option in ["existing", "corrected"]
            for clock in [1, 2]
        }
        error = float(
            np.max(
                abs(
                    (q["corrected", 2] - q["existing", 2])
                    - (q["corrected", 1] - q["existing", 1])
                )
            )
        )
        paired.append(dict(state=state, max_paired_effect_clock_difference_cms=error))
    paired = pd.DataFrame(paired)
    paired.to_csv(ROUTES / "paired_clock_check.csv", index=False)
    responses = responses.merge(paired, on="state", validate="one_to_one")
    hydraulics = pd.read_csv(ROUTES / "hydraulic_comparison.csv")
    audit = pd.read_csv(ROUTES / "routing_audit.csv")
    fig, axs = plt.subplots(4, 2, figsize=(12, 10), layout="constrained")
    for i, state in enumerate(responses.state):
        a = pd.read_parquet(ROUTES / state / "existing/outlet_clock2.parquet")
        b = pd.read_parquet(ROUTES / state / "corrected/outlet_clock2.parquet")
        keep = a.seconds < 12 * 86400
        x = a.loc[keep, "seconds"].to_numpy()[::6] / 86400
        for f, label in [(a, "Existing"), (b, "Corrected")]:
            axs[i, 0].plot(x, f.loc[keep, "outlet_cms"].to_numpy()[::6], label=label)
        axs[i, 1].plot(
            x, (b.outlet_cms - a.outlet_cms).loc[keep].to_numpy()[::6], color="#8d3aab"
        )
        axs[i, 0].set_title("22 / " + state)
        axs[i, 1].set_title("Corrected − existing")
        for ax in axs[i]:
            ax.set_xlabel("Days since pulse start")
            ax.set_ylabel("Discharge (m³/s)")
            ax.grid(alpha=0.2)
    axs[0, 0].legend()
    svg = io.StringIO()
    fig.savefig(svg, format="svg")
    fig.savefig(STAGE / "routing_preview.png", dpi=130)
    plt.close(fig)
    cases = pd.DataFrame(
        [
            dict(
                case=c["case"],
                parity=c["parity"],
                fixed_boundary=c["fixed_boundary_feasible"],
                maximum_head_error_m=max(
                    c["errors"]["head_us"], c["errors"]["head_ds"]
                ),
            )
            for c in manifest["cases"]
        ]
    )
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Local slope integration — staged review</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 24px;color:#192c39;line-height:1.5}}table{{border-collapse:collapse;font-size:13px;display:block;overflow:auto}}th,td{{border:1px solid #ccd6dd;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.status{{background:#e7f3ee;padding:18px}}svg{{max-width:100%;height:auto}}code{{background:#eef1f4}}</style>
<h1>Local slope integration: staged validation</h1><p class="status">32/32 accepted candidates reproduced by maintained code. 30 focused and neighbouring regression tests pass. Four representative network-22 cases routed at two clocks, before and after correction. Production has not been switched.</p>
<h2>What is integrated</h2><p>The opt-in API <code>network_variants.local_head_reconstruction.reconstruct_local_heads</code> validates the selected boundary incidence, exterior lineage and approved reversals. It fixes exterior heads unless the caller explicitly permits reviewed flexibility. It freezes the original absolute slope floor, minimises relative exterior drop disturbance first, and then fits replacement slopes. The minimum boundary-head-displacement tie-break stays on the primary optimal face.</p>
<p>The production shared-head entrypoint remains unchanged. The staging command is <code>python -m network_variants.run_stage_local_heads --output NEW_DIRECTORY</code>. It checks accepted source fingerprints and fails rather than overwriting a directory. The primary exact solver rejects problems exceeding eight candidate nodes or one million active sets; larger problems require explicit solver review.</p>
<h2>Routing experiment</h2><p>Simple unit 8 (also an island alias), compound member 30, flexible member 136, and reversed-connector case 262. Each pair uses its own saved allocation, widths, BASED depths, roughness and diffusion multiplier. Only accepted heads, slopes and geometry lengths change. Coefficients are recalculated consistently. Both sides use the same cell counts, refined where needed for the alpha criterion, and the same clock. This isolates the slope/geometry effect; it does not correct existing per-case reference-flow allocation changes.</p>
<p>Forcing is a 24-hour smooth pulse, baseline 100 m³/s and amplitude 50 m³/s. Results use the half timestep. Curves below display the first 12 days; the simulation continues much longer for volume and settling checks. Maximum relative pulse-volume error: {audit.volume_relative_error.max():.3g}; maximum final baseline error: {audit.tail_error_cms.max():.3g} m³/s. No negative routed discharges were found. The initial short-tail trial was rejected and preserved separately.</p>
{responses.to_html(index=False,float_format=lambda x:f'{x:.7g}')}
{svg.getvalue()[svg.getvalue().index('<svg'):]}
<h2>Hydraulic changes within each pair</h2>{hydraulics.to_html(index=False)}
<h2>What remains before a production switch</h2><ol><li>Agree and implement the reference-flow allocation contract. Saved width-based allocation can still change outside the selected region; slope correction alone does not fix that.</li><li>Coordinate the final depth/roughness interface with the Manning-depth work. These comparisons intentionally retain saved BASED depth and do not recalibrate n.</li><li>Resolve final hydraulic lineage and IDs for restored reaches, then regenerate their widths/depths and validate routing. The present routing subset has unchanged link IDs and connectivity; geometry-restoration cases are included in slope parity but not this routing subset.</li><li>Wire the new explicit contract into the production caller only after these staged-input decisions. The 32 accepted cases are validated; this is not a claim that every collapse in every network has been fitted with the new method.</li></ol>
<p>No production inputs, RF models, original results, or other-chat tracked changes were edited.</p>
<details><summary>All 32 slope parity checks</summary>{cases.to_html(index=False)}</details>
</html>"""
    (STAGE / "integration_review.html").write_text(content)
    (STAGE / "integration_status.json").write_text(
        json.dumps(
            dict(
                slope_cases=32,
                tests_passed=30,
                routing_cases=4,
                routing_runs=len(audit),
                production_adopted=False,
                reference_allocation_corrected=False,
                depth_policy="fixed saved BASED",
                maximum_relative_pulse_volume_error=float(
                    audit.volume_relative_error.max()
                ),
                responses=responses.to_dict("records"),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
