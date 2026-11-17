"""Independent full-edge KKT checks of the hierarchical objectives.

Unlike the fitter, the primary certificate uses full edge incidence with free
internal nodes, not condensed fixed-to-fixed path constraints.
"""
from pathlib import Path
import sys, json, argparse

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from network_variants.reviewed_geometry import ReviewedGeometry


def incidence(f, free):
    ix = {v: i for i, v in enumerate(free)}
    A = np.zeros((len(f), len(free)))
    for j, r in enumerate(f.itertuples()):
        if r.u in ix:
            A[j, ix[r.u]] += 1
        if r.v in ix:
            A[j, ix[r.v]] -= 1
    return A


def nonnegative_certificate(A, gradient, slack):
    if not len(gradient):
        return dict(residual=0.0, active_constraints=0)
    tight = slack < 1e-8
    norm = float(np.max(abs(gradient), initial=0))
    if norm < 1e-12:
        return dict(
            residual=norm, active_constraints=int(tight.sum()), gradient_norm=norm
        )
    target = gradient / norm
    if tight.any():
        lam = nnls(A[tight].T, target, maxiter=max(1000, 20 * int(tight.sum())))[0]
        residual = float(np.max(abs(A[tight].T @ lam - target), initial=0))
    else:
        residual = float(np.max(abs(target), initial=0))
    return dict(
        residual=residual, active_constraints=int(tight.sum()), gradient_norm=norm
    )


def main(folder):
    folder = Path(folder)
    cases = json.loads((folder / "case_results.json").read_text())
    release = ReviewedGeometry()
    cache = {}
    rows = []
    for c in cases:
        if not c["status"].startswith("verified_"):
            continue
        key = c["case"]
        ex, state = key.split("/")
        d = folder / ex / state
        f = pd.read_parquet(d / "slopes.parquet")
        a = json.loads((d / "audit.json").read_text())["solver"]
        spec = json.loads((d / "contract.json").read_text())
        if ex not in cache:
            cache[ex] = pd.read_parquet(
                release.lookup(ex, "indep", "S000_base")["original_frame"]
            ).set_index("id_link")
        p = cache[ex]
        scope = set(spec["selected_parent_links"])
        outside = p[~p.index.isin(scope)]
        fixed = set(outside.u) | set(outside.v) | set(spec["boundary_nodes"])
        candidates = set(a["primary"]["candidates"])
        internal = (set(f.u) | set(f.v)) - fixed
        floor = np.maximum(f["floor"] * f.link_length_m, 1e-10).to_numpy()
        drop = (f.head_us - f.head_ds).to_numpy()
        slack = drop - floor
        row = dict(case=key)
        if candidates:
            free = sorted(internal | candidates)
            A = incidence(f, free)
            ext = f[f.role.eq("exterior")]
            E = incidence(ext, free)
            refs = p.loc[ext.parent_link.astype(int)]
            D = (refs.head_us - refs.head_ds).to_numpy()
            changes = (ext.head_us - ext.head_ds).to_numpy() - D
            w = (1 / D) / np.max(1 / D)
            gradient = E.T @ (w * w * changes)
            certificate = nonnegative_certificate(A, gradient, slack)
            objective = float(np.sum((changes / D) ** 2))
            if not np.isclose(
                objective, a["primary"]["objective"], rtol=1e-7, atol=1e-12
            ):
                raise ValueError("Primary objective disagreement: " + key)
            if certificate["residual"] > 1e-5:
                raise ValueError(f"Primary full-edge KKT failed: {key} {certificate}")
            row.update(primary=certificate, primary_objective=objective)
        patch = f[f.role.eq("selected_replacement")]
        free = sorted(internal)
        A = incidence(patch, free)
        if free:
            target = patch.link_length_m.to_numpy() * a["reference_slope"]
            J = A / target[:, None]
            residual = (patch.head_us - patch.head_ds).to_numpy() / target - 1
            gradient = J.T @ residual / max(np.linalg.norm(J, 2) ** 2, 1.0)
            # Absolute scaled gradient allows cancellation at an unconstrained optimum.
            if np.max(abs(gradient), initial=0) < 1e-7:
                certificate = dict(
                    residual=float(np.max(abs(gradient), initial=0)),
                    active_constraints=0,
                    unconstrained_stationarity=True,
                )
            else:
                slack_patch = (patch.head_us - patch.head_ds).to_numpy() - np.maximum(
                    patch["floor"] * patch.link_length_m, 1e-10
                ).to_numpy()
                certificate = nonnegative_certificate(A, gradient, slack_patch)
                if certificate["residual"] > 1e-5:
                    raise ValueError(f"Secondary KKT failed: {key} {certificate}")
            row["secondary"] = certificate
        rows.append(row)
    summary = dict(
        cases_checked=len(rows),
        primary_certificates=sum("primary" in r for r in rows),
        secondary_certificates=sum("secondary" in r for r in rows),
        maximum_primary_residual=max(
            [r["primary"]["residual"] for r in rows if "primary" in r] or [0]
        ),
        maximum_secondary_residual=max(
            [r["secondary"]["residual"] for r in rows if "secondary" in r] or [0]
        ),
        results=rows,
    )
    (folder / "independent_objective_checks.json").write_text(
        json.dumps(summary, indent=2)
    )
    print({k: v for k, v in summary.items() if k != "results"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    main(a.folder)
