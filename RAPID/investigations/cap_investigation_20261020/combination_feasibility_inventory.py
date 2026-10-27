"""Topology-only inventory, no child construction, routing or fitting."""
from pathlib import Path
import hashlib
import itertools
import json
import sys

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "RAPID/src"))
from rapid_tools.current_domains import validate_domain

OUT = ROOT / "RAPID/outputs/uncapped_event_window_20261020"
DEST = OUT / "combination_feasibility_inventory"
SCOPES = ROOT / "RAPID/outputs/hydrograph_predictor_scopes_20261014"
PAIR = ROOT / "RAPID/outputs/hydrograph_closed_unit_pairing_20261019"
EXT = ROOT / "RAPID/outputs/hydrograph_closed_unit_extension_20261019"
SOURCES = {}


def read(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"example": str})


def closed_regions(frame):
    """Enumerate all bifurcation-to-confluence closed regions in a parent DAG.

    Every node on any source-sink route must be included; external connections
    are allowed only into the inlet and out of the outlet. Parallel edges count.
    Independently revalidate each accepted membership with production validator.
    """
    g = nx.MultiDiGraph()
    for r in frame.itertuples():
        g.add_edge(int(r.id_us_node), int(r.id_ds_node), key=int(r.id_link))
    assert nx.is_directed_acyclic_graph(g) and nx.is_weakly_connected(g)
    order = list(nx.topological_sort(g))
    rank = {n: i for i, n in enumerate(order)}
    descendants = {n: nx.descendants(g, n) | {n} for n in order}
    ancestors = {n: nx.ancestors(g, n) | {n} for n in order}
    incoming = {n: set(g.predecessors(n)) for n in order}
    outgoing = {n: set(g.successors(n)) for n in order}
    regions = []
    for source in [n for n in order if g.out_degree(n) > 1]:
        for sink in [n for n in order if g.in_degree(n) > 1 and rank[n] > rank[source]]:
            if sink not in descendants[source]:
                continue
            nodes = descendants[source] & ancestors[sink]
            if not outgoing[source] <= nodes or not incoming[sink] <= nodes:
                continue
            interior = nodes - {source, sink}
            if any(not (incoming[n] | outgoing[n]) <= nodes for n in interior):
                continue
            ids = frozenset(k for u, v, k in g.edges(nodes, keys=True) if v in nodes)
            part, bounds = validate_domain(frame, frame[frame.id_link.isin(ids)])
            assert bounds == (source, sink)
            regions.append(dict(source=source, sink=sink, ids=ids, nodes=nodes,
                                parent_links=len(ids),
                                edited_parent_km=float((part.reference_flow_fraction * part.link_length_m).sum()/1000)))
    # The first closed outlet for a given split defines a primitive closed bubble.
    # Longer closed regions with the same inlet are serial envelopes, not new atoms.
    first = {}
    for r in regions:
        first.setdefault(r["source"], r["sink"])
        r["primitive"] = first[r["source"]] == r["sink"]
    return g, regions


def adjacent(g, a, b):
    """Same directed unbranched-connector rule as the existing pair inventory."""
    node = a["sink"]
    seen = set()
    while node != b["source"]:
        if node in seen or g.out_degree(node) != 1:
            return False
        if node != a["sink"] and g.in_degree(node) != 1:
            return False
        seen.add(node)
        node = next(iter(g.out_edges(node)))[1]
    return True


def coverage():
    d = read(OUT/"chain80_tuning/evaluation_input.parquet")
    aug = read(OUT/"chain80_tuning/combined_input.parquet")
    # Read the frozen scientific column list without importing fitting code.
    columns = [
        "log1p__B_ratio_route_length_mean", "domain_size",
        "log1p__B_ratio_route_length_sd", "log1p__B_ratio_width_mean",
        "log1p__B_ratio_width_cv", "log1p__B_ratio_route_entropy",
        "B_parent_dominance_mean", "parent_slope_slow", "length_delta_km"]
    x = d[columns].to_numpy(float)
    a = aug[columns].to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(a).all()
    scale = np.std(x, axis=0)
    assert (scale > 0).all()
    z = (x - x.mean(axis=0))/scale
    za = (a - x.mean(axis=0))/scale
    nearest = NearestNeighbors(n_neighbors=2).fit(z).kneighbors(z)[0][:, 1]
    augdist = NearestNeighbors(n_neighbors=1).fit(z).kneighbors(za)[0][:, 0]
    pd.DataFrame(dict(example=aug.example, state=aug.state, standardized_nearest_existing_distance=augdist)).to_csv(DEST/"existing80_coverage.csv", index=False)
    pd.DataFrame([
        dict(predictor=c, existing_min=float(x[:, i].min()), existing_max=float(x[:, i].max()),
             unique_values=int(d[c].nunique()), augmentation_outside_range=int(((a[:, i]<x[:, i].min())|(a[:, i]>x[:, i].max())).sum()))
        for i, c in enumerate(columns)]).to_csv(DEST/"predictor_ranges.csv", index=False)
    result = dict(existing_rows=len(d), unique_vectors=len(d[columns].drop_duplicates()),
                  augmentation_rows=len(aug),
                  existing_neighbor_distance_quantiles=np.quantile(nearest, [.1, .5, .9, .99]).tolist(),
                  augmentation_neighbor_distance_quantiles=np.quantile(augdist, [.1, .5, .9, .99]).tolist(),
                  augmentation_above_existing_90th_neighbor_distance=int((augdist > np.quantile(nearest, .9)).sum()),
                  interpretation="Descriptive standardized Euclidean distances, not a model validation or proof of space coverage. Global scaling used only for inventory. New unconstructed candidates have no nine-dimensional feature vectors yet.")
    (DEST/"coverage_summary.json").write_text(json.dumps(result, indent=2)+"\n")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    outer = read(PAIR/"eligible_blocks.json")
    mappings = read(SCOPES/"domain_graphs.csv")
    plans = [read(EXT/"routing_plan.json"), read(OUT/"routing_plan.json")]
    jobs = {}
    for plan in plans:
        for case in plan["cases"]:
            jobs.setdefault(case["example"], {}).update({j["state_id"]: j for j in case["jobs"]})
    all_regions, all_pairs, summaries, subsets, saved_audit, chains = [], [], [], [], [], []
    for ex in sorted(outer):
        parent = read(Path(jobs[ex]["S000_base"]["parameters"]))
        g, regions = closed_regions(parent)
        saved = {}
        gm = mappings[mappings.example.eq(ex) & mappings.scope.isin(["U", "C"])]
        for key, group in gm[gm.parent_closed].groupby("parent_graph"):
            frame = read(SCOPES/"graphs"/f"{key}.parquet")
            part, _ = validate_domain(parent, frame)
            # Scope files supply membership, active parent supplies attributes.
            assert np.allclose(part.set_index("id_link").loc[frame.id_link, ["link_length_m","width"]],
                               frame[["link_length_m","width"]], rtol=1e-12, atol=1e-10)
            saved[frozenset(frame.id_link.astype(int))] = group
        for i, r in enumerate(regions):
            r["region"] = f"{ex}_closed_{r['source']}_{r['sink']}"
            r["outer"] = [b["state"] for b in outer[ex] if r["ids"] < set(b["parent_ids"])]
            exact = saved.get(r["ids"])
            r["existing_single_section_states"] = [] if exact is None else sorted(set(exact.loc[exact.child_links.eq(1) & exact.child_closed, "state"]))
            r["existing_domain_states"] = [] if exact is None else sorted(set(exact.state))
            all_regions.append(dict(example=ex, region=r["region"], source=r["source"], sink=r["sink"],
                parent_ids=sorted(r["ids"]), parent_links=r["parent_links"], primitive=r["primitive"],
                inside_outer_compound=r["outer"], edited_parent_km=r["edited_parent_km"],
                existing_domain_states=r["existing_domain_states"],
                existing_single_section_states=r["existing_single_section_states"],
                status="parent topology validated; replacement geometry/hydraulics NOT validated"))
        found = {r["ids"] for r in regions}
        for ids, records in saved.items():
            saved_audit.append(dict(example=ex, states=" | ".join(sorted(set(records.state))),
                                    parent_links=len(ids), rediscovered=ids in found))
            assert ids in found, (ex, records.state.tolist())
        primitives = [r for r in regions if r["primitive"]]
        internal = [r for r in primitives if r["outer"]]
        pair_count = serial_count = 0
        arcs = []
        for a, b in itertools.combinations(internal, 2):
            containers = sorted(set(a["outer"]) & set(b["outer"]))
            if not containers or a["ids"] & b["ids"]:
                continue
            ai = a["nodes"] - {a["source"], a["sink"]}
            bi = b["nodes"] - {b["source"], b["sink"]}
            if ai & b["nodes"] or bi & a["nodes"]:
                continue
            forward, backward = adjacent(g, a, b), adjacent(g, b, a)
            serial = forward or backward
            if serial:
                arcs.append((a["region"], b["region"]) if forward else (b["region"], a["region"]))
            ordered = nx.has_path(g, a["sink"], b["source"]) or nx.has_path(g, b["sink"], a["source"])
            pair_count += 1
            serial_count += int(serial)
            all_pairs.append(dict(example=ex, a=a["region"], b=b["region"],
                enclosing_compound=containers[0], adjacent=serial,
                arrangement="adjacent_serial" if serial else "serial_with_intervening_branching" if ordered else "different_branches",
                edited_parent_km=a["edited_parent_km"]+b["edited_parent_km"],
                existing_single_replacements=bool(a["existing_single_section_states"] and b["existing_single_section_states"]),
                status="nonoverlapping closed parent subdomains; joint child not constructed"))
        chain_set = set(arcs)
        frontier = list(arcs)
        while frontier:
            chain = frontier.pop()
            for a, b in arcs:
                new = (*chain, b)
                if chain[-1] == a and b not in chain and new not in chain_set:
                    chain_set.add(new)
                    frontier.append(new)
        by_name = {r["region"]: r for r in internal}
        for chain in sorted(chain_set):
            parts = [by_name[n] for n in chain]
            assert len(set.union(*(set(r["ids"]) for r in parts))) == sum(len(r["ids"]) for r in parts)
            containers = set.intersection(*(set(r["outer"]) for r in parts))
            assert len(containers) == 1
            chains.append(dict(example=ex, regions=" | ".join(chain), n_subregions=len(chain),
                               enclosing_compound=next(iter(containers)),
                               edited_parent_km=sum(r["edited_parent_km"] for r in parts)))
        # Explicit combinatorial lower bound, not enumeration of all antichains.
        # Leaf primitive regions are pairwise disjoint; subsets can be nonadjacent.
        for block in outer[ex]:
            candidates = [r for r in internal if block["state"] in r["outer"]]
            leaves = [r for r in candidates if not any(q["ids"] < r["ids"] for q in candidates)]
            assert all(not a["ids"] & b["ids"] for a,b in itertools.combinations(leaves,2))
            n = len(leaves)
            subsets.append(dict(example=ex, outer_state=block["state"], n_closed_subregions=len(candidates),
                n_disjoint_leaf_subregions=n, subset_count_at_least_two=str(2**n-n-1),
                leaf_regions=" | ".join(r["region"] for r in leaves),
                qualification="combinatorial possibilities only; includes nonadjacent edits, not validated child geometries"))
        summaries.append(dict(example=ex, outer_blocks=len(outer[ex]), closed_bif_conf_regions=len(regions),
            primitive_closed_regions=len(primitives), internal_primitive_regions=len(internal),
            internal_with_saved_single_replacement=sum(bool(r["existing_single_section_states"]) for r in internal),
            internal_closed_pairs=pair_count, internal_adjacent_pairs=serial_count,
            internal_nonadjacent_pairs=pair_count-serial_count,
            internal_adjacent_chains=len(chain_set),
            open_saved_unit_records=int((gm.scope.eq("U") & ~gm.parent_closed).sum())))
        print(ex, summaries[-1], flush=True)
    pd.DataFrame(all_regions).to_parquet(DEST/"closed_regions.parquet", index=False)
    pd.DataFrame(all_pairs).to_csv(DEST/"within_compound_pairs.csv", index=False)
    pd.DataFrame(chains).to_csv(DEST/"within_compound_adjacent_chains.csv", index=False)
    pd.DataFrame(summaries).to_csv(DEST/"network_summary.csv", index=False)
    pd.DataFrame(subsets).to_csv(DEST/"higher_order_lower_bounds.csv", index=False)
    pd.DataFrame(saved_audit).to_csv(DEST/"saved_domain_reproduction.csv", index=False)
    previous = read(OUT/"physical_context_and_combinations/combination_inventory.csv")
    previous.groupby("example").agg(serial_chains=("states","size"), full_segmented_duplicates=("duplicates_segmented","sum")).to_csv(DEST/"outer_serial_inventory_counts.csv")
    coverage()
    for path, digest in SOURCES.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    manifest = dict(sources=SOURCES, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    closed_regions=len(all_regions), within_compound_pairs=len(all_pairs),
                    outer_serial_chains=len(previous), no_candidate_limit=True,
                    no_child_construction=True, no_routing=True, no_model_fitting=True,
                    production_modified=False,
                    scope="Exhaustive closed bifurcation-confluence parent envelopes; exhaustive pairwise nonoverlapping primitive closed subdomains within existing outer blocks. Higher-order arbitrary subsets reported as lower bounds, not exhaustively enumerated.")
    (DEST/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
