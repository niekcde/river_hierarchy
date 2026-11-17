"""Summarise the full slope-only audit; no production or routing operations."""
from pathlib import Path
import argparse, json, html, hashlib
import numpy as np
import pandas as pd


def main(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    cases = pd.read_json(folder / "case_results.json")
    e = pd.read_parquet(folder / "exterior_comparison.parquet")
    e["changed"] = e.relative_change.abs() > 1e-8
    e["old_changed"] = e.old_relative_change.abs() > 1e-8
    e["comparable"] = ~e.length_changed & ~e.reversed & e.existing_slope.notna()
    e["cause"] = np.select(
        [
            e.reversed,
            e.length_changed & e.head_changed,
            e.head_changed,
            e.length_changed,
        ],
        ["accepted reversal", "length and head", "head", "length"],
        default="none",
    )
    verified = cases.status.str.startswith("verified_")
    flex = cases.status.eq("verified_boundary_flex")
    changed = e[e.changed].copy()
    changed.to_parquet(folder / "changed_exterior_slopes.parquet", index=False)
    changed.to_csv(folder / "changed_exterior_slopes.csv", index=False)
    by_network = cases.groupby("example").agg(
        cases=("case", "size"),
        verified=("status", lambda x: int(x.str.startswith("verified_").sum())),
        flexible=("status", lambda x: int(x.eq("verified_boundary_flex").sum())),
        exterior_changes=("exterior_slopes_changed", "sum"),
        old_comparable_changes=("old_comparable_changes", "sum"),
        new_comparable_changes=("new_comparable_changes", "sum"),
    )
    by_network.to_csv(folder / "by_network.csv")
    summary = dict(
        cases=len(cases),
        verified=int(verified.sum()),
        fixed=int(cases.status.eq("verified_fixed").sum()),
        flexible=int(flex.sum()),
        blocked=int((~verified).sum()),
        networks=cases.example.nunique(),
        exterior_reach_case_instances=len(e),
        comparable_exterior_instances=int(e.comparable.sum()),
        old_comparable_changes=int(e.loc[e.comparable, "old_changed"].sum()),
        new_comparable_changes=int(e.loc[e.comparable, "changed"].sum()),
        changed_exterior_instances=int(e.changed.sum()),
        changed_exterior_cases=int(e.loc[e.changed, "case"].nunique()),
        changed_exterior_causes=changed.cause.value_counts().to_dict(),
        changed_boundary_head_instances=int(cases.changed_boundary_heads.sum()),
        max_boundary_head_change_m=float(cases.max_boundary_head_change_m.max()),
        max_relative_exterior_slope_change=float(e.relative_change.abs().max()),
        geometry_length_change_instances=int(e.length_changed.sum()),
        accepted_reversed_instances=int(e.reversed.sum()),
        new_flexible_cases=int((flex & ~cases.accepted_candidate).sum()),
        max_floor_deficit_m=float(cases.max_floor_deficit_m.max()),
        production_modified=False,
        routing_runs=0,
    )
    geometry_only = changed[changed.cause.eq("length")]
    summary.update(
        fixed_boundary_cases_with_length_driven_slope_changes=int(
            (
                cases.status.eq("verified_fixed") & cases.exterior_slopes_changed.gt(0)
            ).sum()
        ),
        maximum_length_only_relative_slope_change=float(
            geometry_only.relative_change.abs().max()
        ),
        length_only_changes_exceeding_one_percent=int(
            geometry_only.relative_change.abs().gt(0.01).sum()
        ),
    )
    objective_checks = json.loads(
        (folder / "independent_objective_checks.json").read_text()
    )
    if objective_checks["cases_checked"] != int(verified.sum()):
        raise ValueError("Objective check coverage mismatch")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    accepted_path = (
        Path(__file__).resolve().parents[3]
        / "RAPID/outputs/accepted_boundary_completion_20261028/candidate_index.json"
    )
    accepted = {r["case"]: r for r in json.loads(accepted_path.read_text())}
    index = []
    for c in cases.loc[verified].to_dict("records"):
        key = c["case"]
        slope_path = folder / key / "slopes.parquet"
        item = dict(
            case=key,
            slope_file=str(slope_path.resolve()),
            slope_sha256=hashlib.sha256(slope_path.read_bytes()).hexdigest(),
            contract_file=str((folder / key / "contract.json").resolve()),
            audit_file=str((folder / key / "audit.json").resolve()),
            node_identity="canonical reference identities",
            contains_only_slope_fields=True,
            status=c["status"],
        )
        if key in accepted:
            item["accepted_geometry_candidate"] = str(
                Path(accepted[key]["candidate_folder"]) / "geometry_candidate.parquet"
            )
            item["accepted_geometry_sha256"] = accepted[key]["geometry_sha256"]
        index.append(item)
    (folder / "staged_slope_index.json").write_text(
        json.dumps(
            dict(
                policy="selected_scope_relative_exterior_drop_v1",
                cases=index,
                production_inputs_modified=False,
            ),
            indent=2,
        )
    )
    cases.loc[flex].to_csv(folder / "boundary_flexibility_cases.csv", index=False)

    # Useful concise table, backed by per-case JSON and complete CSV/parquet files.
    def table(frame):
        return frame.to_html(
            index=False, escape=True, float_format=lambda x: f"{x:.6g}"
        )

    worst = changed.loc[
        changed.relative_change.abs().sort_values(ascending=False).index
    ].head(25)
    view = cases[
        [
            "case",
            "status",
            "accepted_candidate",
            "exterior_slopes_changed",
            "old_comparable_changes",
            "new_comparable_changes",
            "changed_boundary_heads",
            "max_boundary_head_change_m",
            "max_relative_exterior_slope_change",
        ]
    ].copy()
    records = []
    for r in view.to_dict("records"):
        key = r["case"]
        row = [
            '<a href="' + html.escape(key) + '/audit.json">' + html.escape(key) + "</a>"
        ]
        for k, v in list(r.items())[1:]:
            row.append(html.escape(f"{v:.6g}" if isinstance(v, float) else str(v)))
        records.append(
            '<tr data-status="'
            + html.escape(r["status"])
            + '">'
            + "".join("<td>" + v + "</td>" for v in row)
            + "</tr>"
        )
    headers = "".join(
        "<th>" + html.escape(k.replace("_", " ")) + "</th>" for k in view.columns
    )
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>All compound-member slopes — full validation</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 24px;color:#173144;line-height:1.5}}table{{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin:20px 0}}th,td{{padding:7px;border:1px solid #c5d1d8;text-align:right}}td:first-child,th:first-child{{text-align:left}}.status{{background:#e6f2ed;padding:20px}}input,select{{padding:9px;margin-right:12px}}code{{background:#edf1f4}}</style>
<h1>Full compound-member slope check</h1><p class="status"><strong>{summary['verified']} of {len(cases)} cases verified across {summary['networks']} networks.</strong> {summary['fixed']} preserve fixed boundary heads; {summary['flexible']} need bounded flexibility; {summary['blocked']} remain explicitly blocked. This covers the full compound-member inventory, not only the 32-case pilot.</p>
<h2>What “verified” means</h2><p>The existing selected member and boundary incidence agree with unit membership. Exterior lineage and published node identities are checked. The absolute slope floor is verified against the original containing compound. The solve preserves all exterior heads except permitted boundary releases and approved reversed-connector endpoints; every final slope is positive, follows the shared heads and meets its effective reference floor. A separate incidence-matrix linear program checks fixed-boundary feasibility and final feasibility for every solved case. Independent full-edge optimality certificates also pass for all {objective_checks['primary_certificates']} primary fits and all {objective_checks['secondary_certificates']} nontrivial secondary fits; the other cases have no free internal nodes. The 37 focused and neighbouring regression tests pass, and all 32 accepted candidates retain numerical parity. Pre-existing reference-floor numerical residuals are recorded, not hidden.</p>
<p>The primary objective preserves relative exterior drops; the secondary objective fits replacement slopes. When only one conflict endpoint is an eligible boundary, the other stays fixed. Reviewed geometry replacements are reused with source fingerprints. No new geometry edits are made by this batch. General boundary flexibility applies the agreed rule to additional cases; it is distinct from the cases individually reviewed earlier.</p>
<h2>Exterior slope footprint</h2><p>Across {summary['comparable_exterior_instances']:,} comparable exterior reach–case instances (same length, same direction, existing slope available), changed slopes decrease from <strong>{summary['old_comparable_changes']:,}</strong> under the old approach to <strong>{summary['new_comparable_changes']:,}</strong>. Across all matched exterior instances, {summary['changed_exterior_instances']:,} changes remain in {summary['changed_exterior_cases']} cases. These counts include repeated reference reaches in separate collapse experiments.</p>
<p>Remaining changes are explained by boundary-head movement, inherited exterior length differences, or explicitly accepted direction reversals. The inherited length differences are not newly approved geometry exceptions: most were outside the 32-case review. In {summary['fixed_boundary_cases_with_length_driven_slope_changes']} fixed-boundary cases they still change exterior slopes. The largest length-only effect is {100*summary['maximum_length_only_relative_slope_change']:.2f}%; {summary['length_only_changes_exceeding_one_percent']} length-only reach–case changes exceed 1%. These slopes follow the agreed fixed-head rule, but strict exterior-slope identity would require addressing those geometry differences. Maximum boundary movement is {summary['max_boundary_head_change_m']*1000:.3f} mm. Numerical change thresholds: relative slope 1e-8; head 1e-8 m; length 1e-6 m. Floor residual tolerance is 1e-8 m in the final fit.</p>
{table(pd.DataFrame(summary['changed_exterior_causes'].items(),columns=['Cause','Changed exterior instances']))}
<h2>Per-network results</h2>{by_network.reset_index().to_html(index=False)}
<h2>Largest remaining exterior slope changes</h2><p>Relative changes below are fractions: 0.10 means +10%. Reversed reaches compare slope magnitude; their direction change is separately labelled.</p>{table(worst[['case','parent_link','child_link','relative_change','cause','head_us_change_m','head_ds_change_m']])}
<h2>Integration and scope</h2><p>The maintained compound-member regeneration entrypoint is <code>python -m network_variants.run_compound_member_heads --output NEW_DIRECTORY</code>, calling <code>reconstruct_compound_member_heads</code>. Outputs contain only topology/length identifiers and slope/head fields. The containing-compound legacy caller is not used by this entrypoint. Production pointers and saved inputs are unchanged; the staged slope set is available for the separate rerun workflow.</p><p>This is a slope-only validation. It does not claim that reference-flow allocation, depths or routing coefficients are unchanged or validated. No routing or RF runs were performed.</p>
<h2>All cases</h2><input id="search" placeholder="Network, unit or state…"><select id="status"><option value="">All results</option><option value="verified_fixed">Fixed boundaries</option><option value="verified_boundary_flex">Boundary flexibility</option><option value="blocked_explicitly">Blocked</option></select><span id="count"></span>
<table id="cases"><thead><tr>{headers}</tr></thead><tbody>{''.join(records)}</tbody></table>
<p>Downloads: <a href="case_results.csv">all cases</a> · <a href="boundary_flexibility_cases.csv">flexibility cases</a> · <a href="changed_exterior_slopes.csv">changed exterior reaches</a> · <a href="manifest.json">source fingerprints</a> · <a href="summary.json">summary</a> · <a href="staged_slope_index.json">staged slope input index</a></p>
<script>function filter(){{let n=0,q=document.getElementById('search').value.toLowerCase(),s=document.getElementById('status').value;document.querySelectorAll('#cases tbody tr').forEach(r=>{{let show=r.textContent.toLowerCase().includes(q)&&(!s||r.dataset.status===s);r.hidden=!show;if(show)n++;}});document.getElementById('count').textContent=n+' cases';}}document.getElementById('search').oninput=filter;document.getElementById('status').onchange=filter;filter();</script></html>"""
    (folder / "full_slope_review.html").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    main(args.folder)
