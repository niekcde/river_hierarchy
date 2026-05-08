"""Geometry-first synthetic network enumeration workflows."""

import copy
import gzip
import json
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from synthetic_runs.core import (
    Params,
    RiverNetworkNX,
    _cross_loop_intersect,
    _crosses_intersect,
    _disjoint,
    _grid_values,
    _iter_width_splits_two,
    _json_default,
    _merge_summary_parts,
    admissable,
    k_stats_from_graph,
)

# ------------------------------------------------------------------
# 1) Standalone legality (extracts your method logic to work on lists)
# ------------------------------------------------------------------

def break_is_legal_from_list(
    breaks: List[dict],
    *,
    kind: str,
    bf: str,
    bt: str,
    xb: float,
    xr: float,
) -> bool:
    """
    Geometry-only legality check for adding a candidate break to an existing break list.

    This is essentially your RiverNetworkNX.break_is_legal() logic, but standalone.
    """
    xb = float(xb); xr = float(xr)
    bf = str(bf); bt = str(bt)
    kind = str(kind)

    # local helpers copied from your file (assumed in scope); otherwise import them
    # _disjoint, _cross_loop_intersect, _crosses_intersect

    for b in breaks:
        old_kind = b["kind"]
        old_f = b["from_branch"]
        old_t = b["to_branch"]

        # Endpoint uniqueness rule: disallow shared endpoints on the same branch
        set1 = {(xb, bf), (xr, bt)}
        set2 = {(float(b["xb"]), old_f), (float(b["xr"]), old_t)}
        if len(set1 | set2) < 4:
            return False

        # Cross-cross cannot geometrically intersect (after normalizing endpoints by branch)
        if kind == "cross" and old_kind == "cross":
            # Normalize to a pair of x-values ordered by branch label, as you did
            sorted_s1 = sorted(set1, key=lambda x: x[1])
            sorted_s2 = sorted(set2, key=lambda x: x[1])
            newI = [x[0] for x in sorted_s1]  # [x on A, x on B]
            oldI = [x[0] for x in sorted_s2]
            if _crosses_intersect(tuple(newI), tuple(oldI)):
                return False

        # Loop-loop same branch must be disjoint (per your implementation)
        if kind == "loop" and old_kind == "loop" and bf == old_f:
            newI = (xb, xr)
            oldI = (float(b["xb"]), float(b["xr"]))
            if not _disjoint(newI, oldI):
                return False

        # Cross vs loop: forbidden if the cross endpoint on the loop's branch lies inside the loop interval.
        if kind == "cross" and old_kind == "loop":
            loop_br = old_f
            loop_I = (float(b["xb"]), float(b["xr"]))
            if (bf == loop_br) and _cross_loop_intersect(xb, loop_I):
                return False
            if (bt == loop_br) and _cross_loop_intersect(xr, loop_I):
                return False

        if kind == "loop" and old_kind == "cross":
            loop_br = bf
            newI = (xb, xr)
            oldI = (float(b["xb"]), float(b["xr"]))
            if (loop_br == old_f) and _cross_loop_intersect(oldI[0], newI):
                return False
            if (loop_br == old_t) and _cross_loop_intersect(oldI[1], newI):
                return False

    return True


# Optional: keep the class method, but route it through the standalone function.
# (If you paste this into the same file, you can monkey-patch or just edit the class.)
def _patch_break_is_legal_method():
    try:
        RiverNetworkNX.break_is_legal = lambda self, kind, bf, bt, xb, xr: break_is_legal_from_list(
            self.breaks, kind=kind, bf=bf, bt=bt, xb=xb, xr=xr
        )
    except NameError:
        # RiverNetworkNX not in scope; ignore
        pass


# ------------------------------------------------------------------
# 2) Geometry recipe format
# ------------------------------------------------------------------

def geom_meta_from_params(p) -> dict:
    # Keep compatible meta fields
    return dict(
        L=p.L, W_total=p.W_total, xs=p.xs, xe=p.xe,
        jump=p.jump, max_breaks=p.max_breaks,
        min_width=p.min_width, width_step=p.width_step,
        x_stability=p.x_stability,
        Y0=p.Y0, amp_corr=p.amp_corr, amp_loop=p.amp_loop,
    )


def make_geom_recipe(p, breaks: List[dict]) -> dict:
    """
    Geometry-only recipe (no widths anywhere).
    """
    return {"meta": geom_meta_from_params(p), "breaks": breaks}


# ------------------------------------------------------------------
# 3) Candidate break generator (pure geometry)
# ------------------------------------------------------------------

def iter_candidate_breaks(p, *, min_span: Optional[float] = None) -> Iterator[dict]:
    """
    Generate all possible geometric breaks on the grid:
      - loops on A or B
      - crosses A->B and B->A

    Each candidate has no widths.
    """
    if min_span is None:
        min_span = float(p.jump)

    xs_grid = _grid_values(p.xs + p.jump, p.xe, p.jump)

    for xb in xs_grid:
        xr_candidates = _grid_values(xb + p.jump, p.xe, p.jump)
        for xr in xr_candidates:
            if xr <= xb:
                continue
            if (xr - xb) < min_span - 1e-12:
                continue

            # loops
            for br in ("A", "B"):
                yield dict(
                    kind="loop",
                    from_branch=br,
                    to_branch=br,
                    xb=float(xb),
                    xr=float(xr),
                    replace_corridor=True,
                )

            # crosses (both directions)
            yield dict(
                kind="cross",
                from_branch="A",
                to_branch="B",
                xb=float(xb),
                xr=float(xr),
            )
            yield dict(
                kind="cross",
                from_branch="B",
                to_branch="A",
                xb=float(xb),
                xr=float(xr),
            )


def break_sort_key(b: dict) -> tuple:
    """
    Canonical ordering key to prevent generating permutations of the same set.
    """
    return (
        b.get("kind", ""),
        b.get("from_branch", ""),
        b.get("to_branch", ""),
        float(b.get("xb", 0.0)),
        float(b.get("xr", 0.0)),
        bool(b.get("replace_corridor", True)),
    )


# ------------------------------------------------------------------
# 4) Geometry-only DFS enumerator (streamed; bounded RAM)
# ------------------------------------------------------------------

def enumerate_geometric_recipes_streamed(
    p,
    *,
    out_dir: str | Path,
    min_span: Optional[float] = None,
    dedup_sets: bool = True,
    store_set_hash: bool = True,
):
    """
    Enumerate all purely geometric break-sets (up to max_breaks) and write to:
      out_dir/geometry_recipes.jsonl.gz

    - No widths.
    - No NetworkX.
    - Discards only illegal geometry + too-short spans.
    - Uses canonical ordering to avoid permutations.
    - Optionally dedups by set-hash (useful if you later change ordering rules).

    Returns: dict with file path + counts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "run_meta_geometry.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(geom_meta_from_params(p), f, indent=2, default=_json_default)

    out_path = out_dir / "geometry_recipes.jsonl.gz"
    if out_path.exists():
        out_path.unlink()

    candidates = sorted(list(iter_candidate_breaks(p, min_span=min_span)), key=break_sort_key)

    # Optional dedup of sets (break order shouldn't matter)
    seen_sets = set()

    def _set_token(breaks: List[dict]):
        # represent as sorted tuples (order invariant)
        rep = tuple(sorted((break_sort_key(b) for b in breaks)))
        if not store_set_hash:
            return rep
        import hashlib
        payload = json.dumps(rep, separators=(",", ":"), default=_json_default).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).digest()

    n_written = 0
    n_tested_nodes = 0

    with gzip.open(out_path, "wt", encoding="utf-8") as gz:

        def dfs(current: List[dict], start_idx: int):
            nonlocal n_written, n_tested_nodes

            # Every prefix is a valid geometry network
            n_tested_nodes += 1

            if dedup_sets:
                tok = _set_token(current)
                if tok in seen_sets:
                    return
                seen_sets.add(tok)

            gz.write(json.dumps(make_geom_recipe(p, current), default=_json_default) + "\n")
            n_written += 1

            if len(current) >= int(p.max_breaks):
                return

            # Add next breaks in nondecreasing canonical order to avoid permutations
            for i in range(start_idx, len(candidates)):
                cand = candidates[i]

                if not break_is_legal_from_list(
                    current,
                    kind=cand["kind"],
                    bf=cand["from_branch"],
                    bt=cand["to_branch"],
                    xb=cand["xb"],
                    xr=cand["xr"],
                ):
                    continue

                current.append(cand)
                dfs(current, i + 1)
                current.pop()

        dfs([], 0)

    return {
        "meta": str(meta_path),
        "recipes": str(out_path),
        "n_written": int(n_written),
        "n_tested_nodes": int(n_tested_nodes),
        "n_candidates": int(len(candidates)),
        "dedup_sets_size": int(len(seen_sets)) if dedup_sets else None,
    }


# ------------------------------------------------------------------
# 5) Realize widths from a geometry recipe (reuses your RiverNetworkNX)
# ------------------------------------------------------------------

def iter_initial_splits(p) -> Iterator[Tuple[float, float]]:
    for WA in np.arange(p.min_width, p.W_total, p.width_step, dtype=float):
        WB = float(p.W_total - WA)
        if WB < p.min_width - 1e-9:
            continue
        yield float(WA), float(WB)


def iter_realized_networks_from_geom_recipe(
    p,
    geom_recipe: dict,
) -> Iterator["RiverNetworkNX"]:
    """
    For one geometry recipe (no widths), enumerate all width-realizations that satisfy
    your conservation + min_width constraints (via RiverNetworkNX.add_* + recompute_widths).

    IMPORTANT:
    - This can still be large, but it's now conditioned on a single geometry pattern.
    """
    breaks = geom_recipe.get("breaks", [])

    # We apply breaks sequentially and branch only on width splits.
    # This is basically your current DFS, but with fixed break positions/types.
    # print('b1', p)
    # print(geom_recipe)
    for WA0, WB0 in iter_initial_splits(p):
        # print('b2')
        base = RiverNetworkNX(p)
        try:
            base.instantiate_corridor(WA0, WB0)
        except Exception:
            continue
        # print('b3')
        def dfs_apply(net: "RiverNetworkNX", idx: int) -> Iterator["RiverNetworkNX"]:
            if idx >= len(breaks):
                yield net
                return

            b = breaks[idx]
            kind = b["kind"]
            bf = b["from_branch"]
            bt = b["to_branch"]
            xb = float(b["xb"])
            xr = float(b["xr"])

            # compute incoming width at xb on from-branch by doing a temp split
            tmp = copy.deepcopy(net)
            try:
                nb_tmp = tmp.split_corridor_at(bf, xb)
                tmp.recompute_widths()
                W_in = tmp.corridor_incoming_width_at(bf, nb_tmp)
            except Exception:
                return

            if kind == "cross":
                # enumerate W_cross choices as splits of W_in
                for W_rem, W_cross in _iter_width_splits_two(W_in, p.min_width, p.width_step):
                    new = copy.deepcopy(net)
                    try:
                        new.add_cross(bf=bf, bt=bt, xb=xb, xr=xr, W_cross=W_cross)
                    except Exception:
                        continue
                    yield from dfs_apply(new, idx + 1)

            elif kind == "loop":
                replace_corridor = bool(b.get("replace_corridor", True))
                for W1, W2 in _iter_width_splits_two(W_in, p.min_width, p.width_step):
                    new = copy.deepcopy(net)
                    try:
                        new.add_loop(branch=bf, xb=xb, xr=xr, W1=W1, W2=W2, replace_corridor=replace_corridor)
                    except Exception:
                        continue
                    yield from dfs_apply(new, idx + 1)

            else:
                return

        yield from dfs_apply(base, 0)
        # print('b4')

# ------------------------------------------------------------------
# 6) Stream realization of ALL geometry recipes into (recipes + summary)
# ------------------------------------------------------------------

def realize_from_geometry_recipes_streamed(
    p,
    *,
    geometry_recipes_gz: str | Path,
    out_dir: str | Path,
    summary_format: str = "parquet",   # "parquet" or "csv"
    filter_k_admissible: bool = True,
    rows_chunk: int = 50000,
    recipe_chunk: int = 5000,
):
    """
    Reads geometry recipes (jsonl.gz), realizes widths (enumerates all feasible width assignments),
    and writes:
      - out_dir/networks.jsonl.gz          (realized recipes WITH widths; compatible with your loader)
      - out_dir/summary.parquet or .csv    (realized summaries)
      - out_dir/run_meta_realization.json

    Returns: (paths_dict, merged_summary_df)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "run_meta_realization.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(geom_meta_from_params(p), f, indent=2, default=_json_default)

    recipes_out = out_dir / "networks.jsonl.gz"
    if recipes_out.exists():
        recipes_out.unlink()

    parts_dir = out_dir / "summary_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    rows_buf: List[dict] = []
    recipe_buf: List[str] = []
    part_idx = 0
    network_id = 0

    tested_by_breaks = {d: 0 for d in range(p.max_breaks + 1)}
    kept_by_breaks = {d: 0 for d in range(p.max_breaks + 1)}

    def dump_rows_part():
        nonlocal part_idx, rows_buf
        if not rows_buf:
            return
        df_part = pd.DataFrame(rows_buf)
        suffix = "parquet" if summary_format.lower() == "parquet" else "csv"
        part_path = parts_dir / f"summary_part_{part_idx:05d}.{suffix}"
        if summary_format.lower() == "parquet":
            df_part.to_parquet(part_path, index=False)
        else:
            df_part.to_csv(part_path, index=False)
        rows_buf = []
        part_idx += 1

    def dump_recipe_batch(gz):
        nonlocal recipe_buf
        if not recipe_buf:
            return
        gz.write("\n".join(recipe_buf) + "\n")
        recipe_buf = []

    # Stream output gzip once
    with gzip.open(recipes_out, "wt", encoding="utf-8") as gz_out:
        with gzip.open(geometry_recipes_gz, "rt", encoding="utf-8") as gz_in:
            for line in gz_in:
                geom = json.loads(line)
                breaks = geom.get("breaks", [])
                depth = len(breaks)
                if depth > p.max_breaks:
                    continue

                # Realize all width variants for this geometry
                for net in iter_realized_networks_from_geom_recipe(p, geom):
                    tested_by_breaks[depth] += 1

                    ks = k_stats_from_graph(net.G, p.x_stability)
                    if ks is None:
                        continue
                    if filter_k_admissible and (not ks.get("admissible", False)):
                        continue

                    try:
                        n_paths = net.count_paths()
                    except Exception:
                        continue

                    kept_by_breaks[depth] += 1

                    rows_buf.append({
                        "network_id": network_id,
                        "n_breaks": depth,
                        "n_paths": int(n_paths),
                        "WA0": float(net.WA0),
                        "WB0": float(net.WB0),
                        **ks,
                    })

                    recipe_buf.append(json.dumps(net.to_recipe(), default=_json_default))
                    network_id += 1

                    if len(recipe_buf) >= int(recipe_chunk):
                        dump_recipe_batch(gz_out)
                    if len(rows_buf) >= int(rows_chunk):
                        dump_rows_part()

    # final flush
    if recipe_buf:
        with gzip.open(recipes_out, "at", encoding="utf-8") as gz_out:
            gz_out.write("\n".join(recipe_buf) + "\n")
        recipe_buf = []

    if rows_buf:
        dump_rows_part()

    # merge parts to final summary
    merged = _merge_summary_parts(parts_dir, out_dir, summary_format.lower())

    # add run-level columns like your original
    if not merged.empty:
        merged["n_tested_total"] = int(sum(tested_by_breaks.values()))
        merged["tested_n_breaks"] = merged["n_breaks"].map(tested_by_breaks)
        merged["admissible_n_breaks"] = merged["n_breaks"].map(kept_by_breaks)

        merged["admissable_0.1"] = admissable(merged["k_ratio"], 0.1)
        merged["admissable_0.2"] = admissable(merged["k_ratio"], 0.2)
        merged["admissable_0.3"] = admissable(merged["k_ratio"], 0.3)

        if summary_format.lower() == "parquet":
            merged.to_parquet(out_dir / "summary.parquet", index=False)
        else:
            merged.to_csv(out_dir / "summary.csv", index=False)

    # cleanup temp parts
    shutil.rmtree(parts_dir, ignore_errors=True)

    return {
        "meta": str(meta_path),
        "networks": str(recipes_out),
        "summary": str(out_dir / ("summary.parquet" if summary_format.lower() == "parquet" else "summary.csv")),
    }, merged


def main():
    """Run the legacy standalone geometry-enumeration example."""
    _patch_break_is_legal_method()
    maxW = 400
    p = Params(
        L=10000,
        W_total=maxW,
        xs=3000,
        xe=7000,
        jump=500,
        max_breaks=6,
        min_width=10,
        width_step=10,
        x_stability=0.1,
    )

    out_geom = f"/Volumes/PhD/river_hierarchy/output/synthetic_network/synthetic_run_{p.max_breaks}_geom"
    info = enumerate_geometric_recipes_streamed(p, out_dir=out_geom, min_span=p.jump)
    print("GEOM:", info)

    # Then optionally realize widths (this can still be big, but now conditioned on geometry file)
    # out_real = "out_realized_run"
    # paths, summary = realize_from_geometry_recipes_streamed(
    #     p,
    #     geometry_recipes_gz=Path(out_geom) / "geometry_recipes.jsonl.gz",
    #     out_dir=out_real,
    #     summary_format="parquet",
    #     filter_k_admissible=True,   # set False if you truly want geometry+width only
    # )
    # print("REALIZED paths:", paths)
    # print("REALIZED summary head:\n", summary.head())


__all__ = [
    "break_is_legal_from_list",
    "geom_meta_from_params",
    "make_geom_recipe",
    "iter_candidate_breaks",
    "break_sort_key",
    "enumerate_geometric_recipes_streamed",
    "iter_initial_splits",
    "iter_realized_networks_from_geom_recipe",
    "realize_from_geometry_recipes_streamed",
    "main",
]


if __name__ == "__main__":
    main()
