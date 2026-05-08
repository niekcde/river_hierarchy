"""Width-sampling workflows for realized synthetic networks."""

import copy
import gzip
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_runs.core import (
    Params,
    RiverNetworkNX,
    _edge_uid,
    _json_default,
    canonical_signature,
    compute_k,
    ebi,
    edges_spanning_x,
    k_stats_from_graph,
    x_midpoints,
)
# ------------------------------------------------------------
# EBI min/mean/max (uses your existing helpers x_midpoints, edges_spanning_x, ebi)
# ------------------------------------------------------------

def ebi_stats_min_mean_max(G) -> Dict[str, float]:
    """
    Compute EBI at midpoints like your original metrics_by_midpoint,
    but return min/mean/max. Uses width attribute 'w'.
    Ignores same-x edges automatically because they don't span any midpoint.
    """
    xs, mids = x_midpoints(G, x_attr="x")
    vals = []
    bi = []
    # your original uses mids[1:-1]; keep same convention
    for x0 in mids[1:-1]:
        spanning = edges_spanning_x(G, x0)
        if not spanning:
            continue
        ws = np.array([float(data.get("w", 0.0)) for _, _, data in spanning], dtype=float)
        ws = ws[ws > 0]
        if ws.size == 0:
            continue
        vals.append(float(ebi(ws)))
        bi.append(len(spanning))

    if not vals:
        return {"ebi_min": float("nan"), "ebi_mean": float("nan"), "ebi_max": float("nan"), 'bi':float("nan")}

    arr = np.array(vals, dtype=float)
    return {"ebi_min": float(arr.min()), "ebi_mean": float(arr.mean()), "ebi_max": float(arr.max()), "bi": np.mean(bi)}


# ------------------------------------------------------------
# Edge-level stats table
# ------------------------------------------------------------

def edge_stats_rows(net, network_id: int, geometry_id: int, sample_id: int) -> List[dict]:
    """
    One row per edge, including per-edge k, width, length.
    Includes main edges and corridor edges, and loop/cross edges.
    Same-x connector edges will have length=0 and k=NaN.
    """
    G = net.G
    rows = []
    for u, v, k, d in G.edges(keys=True, data=True):
        xu = float(G.nodes[u]["x"])
        xv = float(G.nodes[v]["x"])
        L = xv - xu
        w = float(d.get("w", 0.0))
        kind = d.get("kind", "unknown")

        kval = float("nan")
        if (L > 0) and (w > 0):
            kval = float(compute_k(L, w))

        rows.append({
            "network_id": int(network_id),
            "geometry_id": int(geometry_id),
            "sample_id": int(sample_id),
            "edge_uid": _edge_uid(u, v, k),
            "u": str(u),
            "v": str(v),
            "key": str(k),
            "kind": kind,
            "branch": d.get("branch", None),
            "from_branch": d.get("from_branch", None),
            "to_branch": d.get("to_branch", None),
            "x_u": xu,
            "x_v": xv,
            "length": float(L),
            "width": float(w),
            "k": kval,
        })
    return rows


# ------------------------------------------------------------
# Summary row for a realized network
# ------------------------------------------------------------

def network_summary_row(net, network_id: int, geometry_id: int, sample_id: int, is_benchmark: bool) -> dict:
    """
    One row per realized network. Adds:
      - k stats (existing)
      - ebi min/mean/max
      - n_paths
    """
    ks = k_stats_from_graph(net.G, net.p.x_stability)  # uses widths already assigned
    if ks is None:
        raise ValueError("No k-stats (no valid edges)")

    # paths
    n_paths = net.count_paths()

    # EBI min/mean/max
    ebi_stats = ebi_stats_min_mean_max(net.G)

    row = {
        "network_id": int(network_id),
        "geometry_id": int(geometry_id),
        "sample_id": int(sample_id),
        "is_benchmark": bool(is_benchmark),
        "n_breaks": int(len(net.breaks)),
        "n_paths": int(n_paths),
        "WA0": float(net.WA0),
        "WB0": float(net.WB0),
        **ks,
        **ebi_stats,
    }
    return row


# ------------------------------------------------------------
# Sampling utilities
# ------------------------------------------------------------
RATIO_WEIGHTS = [0.3, 0.8, 1, 1, 1]  # uniform
_RATIO_OPTIONS = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

def _pick_ratio_pair(rng: random.Random, allowed=_RATIO_OPTIONS) -> Tuple[float, float]:
    return rng.choices(list(allowed), weights = RATIO_WEIGHTS, k = 1)[0]

def pick_ratio_pair_feasible(rng, W_in, min_width, allowed=_RATIO_OPTIONS, weights=None):
    feasible = []
    feasible_w = []
    for i, (a, b) in enumerate(allowed):
        # both shares must be >= min_width
        if (W_in * a >= min_width - 1e-9) and (W_in * b >= min_width - 1e-9):
            feasible.append((a, b))
            if weights is None:
                feasible_w.append(1.0)
            else:
                feasible_w.append(weights[i])

    if not feasible:
        return None  # caller decides what to do (raise or fallback)

    # random.choices does weighted sampling
    return rng.choices(feasible, weights=feasible_w, k=1)[0]

def _apply_flip(pair: Tuple[float, float], flip: bool) -> Tuple[float, float]:
    a, b = pair
    return (b, a) if flip else (a, b)

@dataclass
class SplitChoice:
    ratio: Tuple[float, float]   # (r_big, r_small) before flip; we store chosen pair pre-flip too
    flip: bool                   # whether to swap
    # interpretation depends on event type (initial, loop, cross)
    # for cross: first component after flip will be W_cross (by our convention below)

@dataclass
class SamplePlan:
    initial: SplitChoice
    per_break: List[SplitChoice]  # aligned with geom_recipe["breaks"]


def make_benchmark_plan(n_breaks: int) -> SamplePlan:
    c = SplitChoice(ratio=(0.5, 0.5), flip=False)
    return SamplePlan(initial=c, per_break=[c for _ in range(n_breaks)])


def make_random_plan(rng: random.Random, n_breaks: int,
                     allowed=_RATIO_OPTIONS) -> SamplePlan:
    init_pair = _pick_ratio_pair(rng, allowed)
    init_flip = rng.choice([False, True])
    initial = SplitChoice(ratio=init_pair, flip=init_flip)

    per = []
    for _ in range(n_breaks):
        pair = _pick_ratio_pair(rng, allowed)
        flip = rng.choice([False, True])
        per.append(SplitChoice(ratio=pair, flip=flip))

    return SamplePlan(initial=initial, per_break=per)


# ------------------------------------------------------------
# Realize a single geometry recipe under a specific SamplePlan
# ------------------------------------------------------------

# def realize_geom_with_plan(p, geom_recipe: dict, plan: SamplePlan) -> "RiverNetworkNX":
    """
    Build one RiverNetworkNX with widths determined by percentages in plan.
    Enforces min_width on every created edge & on remaining corridor width after crosses.
    Uses your existing add_loop/add_cross which call recompute_widths().
    """
    breaks = geom_recipe.get("breaks", [])

    # Initial split at xs: W_total -> (WA0, WB0)
    a, b = _apply_flip(plan.initial.ratio, plan.initial.flip)
    WA0 = float(p.W_total * a)
    WB0 = float(p.W_total * b)

    if WA0 < p.min_width - 1e-9 or WB0 < p.min_width - 1e-9:
        raise ValueError("Initial split violates min_width")

    net = RiverNetworkNX(p)
    net.instantiate_corridor(WA0, WB0)

    # Apply each break with its plan choice
    for idx, brk in enumerate(breaks):
        choice = plan.per_break[idx]
        kind = brk["kind"]
        bf = brk["from_branch"]
        bt = brk["to_branch"]
        xb = float(brk["xb"])
        xr = float(brk["xr"])
        replace_corridor = bool(brk.get("replace_corridor", True))

        # Determine incoming width W_in at xb on from-branch
        tmp = copy.deepcopy(net)
        nb_tmp = tmp.split_corridor_at(bf, xb)
        tmp.recompute_widths()
        W_in = float(tmp.corridor_incoming_width_at(bf, nb_tmp))

        if W_in < p.min_width - 1e-9:
            raise ValueError("Incoming corridor width already below min_width")

        if kind == "cross":
            # Convention: after flip, first component is the cross share
            r_cross, r_rem = _apply_flip(choice.ratio, choice.flip)
            W_cross = float(W_in * r_cross)
            W_rem = float(W_in - W_cross)

            # Must satisfy min widths for cross edge and remaining corridor
            if W_cross < p.min_width - 1e-9:
                raise ValueError("Cross width violates min_width")
            if W_rem < p.min_width - 1e-9:
                raise ValueError("Remaining corridor width after cross violates min_width")

            net.add_cross(bf=bf, bt=bt, xb=xb, xr=xr, W_cross=W_cross)

        elif kind == "loop":
            # Loop splits into two parallel edges; must both satisfy min_width
            r1, r2 = _apply_flip(choice.ratio, choice.flip)
            W1 = float(W_in * r1)
            W2 = float(W_in - W1)

            if W1 < p.min_width - 1e-9 or W2 < p.min_width - 1e-9:
                raise ValueError("Loop widths violate min_width")

            # Loop replaces corridor segment; your add_loop expects W1+W2 == W_in (true)
            net.add_loop(branch=bf, xb=xb, xr=xr, W1=W1, W2=W2, replace_corridor=replace_corridor)

        else:
            raise ValueError(f"Unknown break kind: {kind}")

    return net

def realize_geom_with_plan(p, geom_recipe: dict, plan: SamplePlan, rng: random.Random,
                           ratio_weights=None, allowed=_RATIO_OPTIONS) -> Tuple["RiverNetworkNX", SamplePlan]:
    breaks = geom_recipe.get("breaks", [])

    # Initial split stays from plan (so benchmark works)
    a, b = _apply_flip(plan.initial.ratio, plan.initial.flip)
    WA0 = float(p.W_total * a)
    WB0 = float(p.W_total * b)
    if WA0 < p.min_width - 1e-9 or WB0 < p.min_width - 1e-9:
        raise ValueError("Initial split violates min_width")

    net = RiverNetworkNX(p)
    net.instantiate_corridor(WA0, WB0)

    # Apply each break
    used_per_break: List[SplitChoice] = []  # ### CHANGED LINES: record actual choices
    for idx, brk in enumerate(breaks):
        kind = brk["kind"]
        bf = brk["from_branch"]
        bt = brk["to_branch"]
        xb = float(brk["xb"])
        xr = float(brk["xr"])
        replace_corridor = bool(brk.get("replace_corridor", True))

        # Determine incoming width W_in at xb on from-branch
        tmp = copy.deepcopy(net)
        nb_tmp = tmp.split_corridor_at(bf, xb)
        tmp.recompute_widths()
        W_in = float(tmp.corridor_incoming_width_at(bf, nb_tmp))

        if W_in < p.min_width - 1e-9:
            raise ValueError("Incoming corridor width already below min_width")

        # ---- NEW: choose ratio conditionally (unless this is benchmark mode)
        # If you want benchmark to force 50/50 at every break, detect it:
        is_benchmark = (plan is not None and plan.per_break and plan.per_break[idx].ratio == (0.5,0.5)
                        and plan.per_break[idx].flip is False and plan.initial.ratio == (0.5,0.5))

        if is_benchmark:
            pair = (0.5, 0.5)
            flip = False
        else:
            pair = pick_ratio_pair_feasible(rng, W_in, p.min_width, allowed=allowed, weights=ratio_weights)
            if pair is None:
                raise ValueError("No feasible ratio for this W_in")
            flip = rng.choice([False, True])

        rA, rB = _apply_flip(pair, flip)
        used_per_break.append(SplitChoice(ratio=pair, flip=flip))  # ### CHANGED LINES

        if kind == "cross":
            # convention: rA is cross share
            W_cross = float(W_in * rA)
            W_rem = float(W_in - W_cross)
            if W_cross < p.min_width - 1e-9 or W_rem < p.min_width - 1e-9:
                raise ValueError("Cross violates min_width (shouldn't happen if feasibility check is correct)")
            net.add_cross(bf=bf, bt=bt, xb=xb, xr=xr, W_cross=W_cross)

        elif kind == "loop":
            W1 = float(W_in * rA)
            W2 = float(W_in - W1)
            if W1 < p.min_width - 1e-9 or W2 < p.min_width - 1e-9:
                raise ValueError("Loop violates min_width (shouldn't happen if feasibility check is correct)")
            net.add_loop(branch=bf, xb=xb, xr=xr, W1=W1, W2=W2, replace_corridor=replace_corridor)
        else:
            raise ValueError(f"Unknown break kind: {kind}")

    used_plan = SamplePlan(initial=plan.initial, per_break=used_per_break)  # ### CHANGED LINES
    return net, used_plan  # ### CHANGED LINES


# ------------------------------------------------------------
# Dedup signature within each geometry's 5 samples
# ------------------------------------------------------------

def realized_signature(net) -> object:
    """
    A/B-invariant signature. Reuses your existing canonical_signature(net).
    For speed/memory, hash it if needed.
    """
    sig = canonical_signature(net)
    # Optional hashing to reduce memory; uncomment if needed:
    # import hashlib
    # payload = json.dumps(sig, separators=(",", ":"), default=_json_default).encode("utf-8")
    # return hashlib.blake2b(payload, digest_size=16).digest()
    return sig


# ------------------------------------------------------------
# Stream-read geometry recipes with geometry_id = line number
# ------------------------------------------------------------

def iter_geometry_recipes_with_id(geometry_recipes_gz: str | Path) -> Iterator[Tuple[int, dict]]:
    with gzip.open(geometry_recipes_gz, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield i, json.loads(line)


# ------------------------------------------------------------
# Chunked parquet writers (bounded RAM)
# ------------------------------------------------------------

def _write_part(df: pd.DataFrame, parts_dir: Path, prefix: str, part_idx: int, fmt: str):
    parts_dir.mkdir(parents=True, exist_ok=True)
    suffix = "parquet" if fmt == "parquet" else "csv"
    path = parts_dir / f"{prefix}_part_{part_idx:05d}.{suffix}"
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

def _merge_parts(parts_dir: Path, out_path: Path, prefix: str, fmt: str) -> pd.DataFrame:
    suffix = "parquet" if fmt == "parquet" else "csv"
    parts = sorted(parts_dir.glob(f"{prefix}_part_*.{suffix}"))
    if not parts:
        df = pd.DataFrame()
        if fmt == "parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        return df

    dfs = []
    for pth in parts:
        if fmt == "parquet":
            dfs.append(pd.read_parquet(pth))
        else:
            dfs.append(pd.read_csv(pth))
    df = pd.concat(dfs, ignore_index=True)
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return df


# ------------------------------------------------------------
# MAIN DRIVER
# ------------------------------------------------------------

def sample_realized_networks_from_geometry(
    *,
    params_json: str | Path,
    geometry_recipes_gz: str | Path,
    out_dir: str | Path,
    n_samples: int = 5,                 # 1 benchmark + (n_samples-1) random
    ratios: List[Tuple[float, float]] = None,
    seed: int = 123,
    max_attempts_per_sample: int = 500,
    filter_k_admissible: bool = True,   # your k_ratio admissibility
    summary_format: str = "parquet",    # "parquet" or "csv"
    write_edges: bool = True,           # edges.parquet for QA
    rows_chunk: int = 50000,
    W_total: float | None = None,
    min_width: float | None = None,
):
    """
    Inputs:
      - params_json: your Params metadata file (run_meta_geometry.json or equivalent)
      - geometry_recipes_gz: geometry_recipes.jsonl.gz
      - W_total: override max width (aka W_total) from params_json
      - min_width: override min_width from params_json
    Outputs in out_dir:
      - networks.jsonl.gz (realized recipes with sampling info)
      - summary.parquet/csv (network-level)
      - edges.parquet/csv (optional)
      - run_meta_sampling.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load params
    with open(params_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta = dict(meta)
    if W_total is not None:
        meta["W_total"] = float(W_total)
    if min_width is not None:
        meta["min_width"] = float(min_width)
    p = Params(**{k: meta[k] for k in Params.__dataclass_fields__.keys()})

    # ratio set
    if ratios is None:
        ratios = _RATIO_OPTIONS
    else:
        ratios = [(float(a), float(b)) for a, b in ratios]

    rng = random.Random(seed)

    # write run meta
    run_meta = {
        "params": meta,
        "n_samples": n_samples,
        "ratios": ratios,
        "seed": seed,
        "max_attempts_per_sample": max_attempts_per_sample,
        "filter_k_admissible": filter_k_admissible,
    }
    with open(out_dir / "run_meta_sampling.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, default=_json_default)

    # output files
    networks_out = out_dir / "networks.jsonl.gz"
    if networks_out.exists():
        networks_out.unlink()

    summary_parts = out_dir / "summary_parts"
    edges_parts = out_dir / "edges_parts"
    shutil.rmtree(summary_parts, ignore_errors=True)
    shutil.rmtree(edges_parts, ignore_errors=True)
    summary_parts.mkdir(parents=True, exist_ok=True)
    if write_edges:
        edges_parts.mkdir(parents=True, exist_ok=True)

    summary_buf: List[dict] = []
    edges_buf: List[dict] = []
    summary_part_idx = 0
    edges_part_idx = 0

    network_id = 0

    # To help debugging:
    geom_fail_rows = []

    def flush_if_needed():
        nonlocal summary_part_idx, edges_part_idx, summary_buf, edges_buf
        if len(summary_buf) >= rows_chunk:
            _write_part(pd.DataFrame(summary_buf), summary_parts, "summary", summary_part_idx, summary_format)
            summary_buf = []
            summary_part_idx += 1
        if write_edges and len(edges_buf) >= rows_chunk:
            _write_part(pd.DataFrame(edges_buf), edges_parts, "edges", edges_part_idx, summary_format)
            edges_buf = []
            edges_part_idx += 1

    with gzip.open(networks_out, "wt", encoding="utf-8") as gz_out:
        for geometry_id, geom in iter_geometry_recipes_with_id(geometry_recipes_gz):
            breaks = geom.get("breaks", [])
            n_breaks = len(breaks)

            # Skip geometries that touch xs/xe endpoints if your model can't realize them
            # (optional safety; comment out if you already fixed generator)
            bad = False
            for b in breaks:
                xb = float(b["xb"]); xr = float(b["xr"])
                if abs(xb - p.xs) < 1e-9 or abs(xb - p.xe) < 1e-9 or abs(xr - p.xs) < 1e-9 or abs(xr - p.xe) < 1e-9:
                    bad = True
                    break
            if bad:
                geom_fail_rows.append({"geometry_id": geometry_id, "reason": "touches_xs_or_xe"})
                continue

            # Per-geometry seen signatures (dedup within the 5)
            seen_sigs = set()

            # Build benchmark + random samples
            target = int(n_samples)
            kept = 0

            # Predefine benchmark plan for sample_id=0
            # For 0-break case, we still want 5 total (benchmark + 4 ratio variants).
            plans_queue: List[Tuple[int, Optional[SamplePlan], str]] = []
            plans_queue.append((0, make_benchmark_plan(n_breaks), "benchmark"))

            # For 0-break special case: enforce 4 deterministic ratio choices (60/40..90/10)
            if n_breaks == 0:
                fixed = [(0.6,0.4),(0.7,0.3),(0.8,0.2),(0.9,0.1)]
                for sid, pair in enumerate(fixed, start=1):
                    # random flip for who gets bigger
                    flip = rng.choice([False, True])
                    c = SplitChoice(ratio=pair, flip=flip)
                    plans_queue.append((sid, SamplePlan(initial=c, per_break=[]), "fixed_zero_break"))
            else:
                # For nonzero breaks: we will generate random plans on demand for sample_id 1..target-1
                for sid in range(1, target):
                    plans_queue.append((sid, None, "random"))

            # Iterate through desired sample slots
            for (sample_id, preplan, mode) in plans_queue:
                if kept >= target:
                    break

                attempts = 0
                accepted = False

                while attempts < max_attempts_per_sample and not accepted:
                    attempts += 1

                    if mode == "benchmark" or mode == "fixed_zero_break":
                        plan = preplan
                    else:
                        plan = make_random_plan(rng, n_breaks, allowed=ratios)

                    try:
                        net, used_plan = realize_geom_with_plan(  # ### CHANGED LINES
                            p, geom, plan, rng=rng, ratio_weights=None, allowed=ratios
                        )

                    except Exception:
                        continue

                    # K admissibility filter (optional)
                    try:
                        ks = k_stats_from_graph(net.G, p.x_stability)
                        if ks is None:
                            continue
                        if filter_k_admissible and not ks.get("admissible", False):
                            continue
                    except Exception:
                        continue

                    # similarity check within geometry
                    try:
                        sig = realized_signature(net)
                    except Exception:
                        continue

                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)

                    # accepted -> write outputs
                    # Attach sampling info to recipe for reproducibility
                    rec = net.to_recipe()
                    rec["geometry_id"] = int(geometry_id)
                    rec["sample_id"] = int(sample_id)
                    rec["sample_mode"] = mode
                    rec["sample_plan"] = {
                        "initial": {"ratio": list(used_plan.initial.ratio), "flip": bool(used_plan.initial.flip)},  # ### CHANGED LINES
                        "per_break": [{"ratio": list(c.ratio), "flip": bool(c.flip)} for c in used_plan.per_break],  # ### CHANGED LINES
                    }

                    gz_out.write(json.dumps(rec, default=_json_default) + "\n")

                    # summary row
                    try:
                        srow = network_summary_row(net, network_id, geometry_id, sample_id, mode == "benchmark")
                        # overwrite EBI with min/mean/max already included; ks includes mean/max but fine
                    except Exception:
                        # if summary fails, don't keep
                        seen_sigs.remove(sig)
                        continue

                    summary_buf.append(srow)

                    # edges table (QA)
                    if write_edges:
                        edges_buf.extend(edge_stats_rows(net, network_id, geometry_id, sample_id))

                    network_id += 1
                    kept += 1
                    accepted = True
                    flush_if_needed()

                if not accepted:
                    geom_fail_rows.append({
                        "geometry_id": int(geometry_id),
                        "sample_id": int(sample_id),
                        "reason": f"max_attempts_exceeded_{mode}",
                        "n_breaks": int(n_breaks),
                    })
                    # Continue to next sample slot; you requested “try to still get 5”
                    # but if a slot can’t be filled, we log it. You can increase max_attempts.

            # If we ended with < target kept, log it
            if kept < target:
                geom_fail_rows.append({
                    "geometry_id": int(geometry_id),
                    "reason": "could_not_fill_all_samples",
                    "kept": int(kept),
                    "target": int(target),
                    "n_breaks": int(n_breaks),
                })

    # final flush buffers
    if summary_buf:
        _write_part(pd.DataFrame(summary_buf), summary_parts, "summary", summary_part_idx, summary_format)
        summary_part_idx += 1
        summary_buf = []
    if write_edges and edges_buf:
        _write_part(pd.DataFrame(edges_buf), edges_parts, "edges", edges_part_idx, summary_format)
        edges_part_idx += 1
        edges_buf = []

    # merge to final outputs
    summary_out = out_dir / ("summary.parquet" if summary_format == "parquet" else "summary.csv")
    summary_df = _merge_parts(summary_parts, summary_out, "summary", summary_format)
    shutil.rmtree(summary_parts, ignore_errors=True)

    edges_df = None
    edges_out = None
    if write_edges:
        edges_out = out_dir / ("edges.parquet" if summary_format == "parquet" else "edges.csv")
        edges_df = _merge_parts(edges_parts, edges_out, "edges", summary_format)
        shutil.rmtree(edges_parts, ignore_errors=True)

    # failures log
    fail_df = pd.DataFrame(geom_fail_rows)
    fail_path = out_dir / "failures.csv"
    fail_df.to_csv(fail_path, index=False)

    return {
        "params_used": str(params_json),
        "geometry_recipes": str(geometry_recipes_gz),
        "networks": str(networks_out),
        "summary": str(summary_out),
        "edges": str(edges_out) if edges_out else None,
        "failures": str(fail_path),
        "n_realized": int(len(summary_df)) if summary_df is not None else 0,
    }
def main():
    """Run the legacy standalone width-sampling example."""
    breaks = 5
    directory = f"/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_{breaks}_geom/"
    W_total_input = 650  # set to a number to override params_json (max width)
    min_width_input = 10  # set to a number to override params_json
    paths = sample_realized_networks_from_geometry(
        params_json=directory + "run_meta_geometry.json",
        geometry_recipes_gz=directory + "geometry_recipes.jsonl.gz",
        out_dir=directory + "sampled_realizations/",
        n_samples=100,
        seed=42,
        max_attempts_per_sample=500,
        filter_k_admissible=False,
        summary_format="parquet",
        write_edges=True,     # for QA
        W_total=W_total_input,
        min_width=min_width_input,
    )
    print(paths)


__all__ = [
    "SplitChoice",
    "SamplePlan",
    "ebi_stats_min_mean_max",
    "edge_stats_rows",
    "network_summary_row",
    "pick_ratio_pair_feasible",
    "make_benchmark_plan",
    "make_random_plan",
    "realize_geom_with_plan",
    "realized_signature",
    "iter_geometry_recipes_with_id",
    "sample_realized_networks_from_geometry",
    "main",
]


if __name__ == "__main__":
    main()
