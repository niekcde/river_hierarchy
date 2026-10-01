"""Baseline construction and planned lineage; collapse execution is separate."""
from dataclasses import dataclass
import math

from .core import Params, RiverNetworkNX

FAMILIES = ("centerline", "simple", "nested", "crossover")


@dataclass(frozen=True)
class Design:
    total_width_m: float = 800.0
    head_drop_m: float = 2.0
    outer_a_fraction: float = 0.8
    inner_fraction: float = 0.5
    total_length_m: float = 10000.0
    internal_location: str = "central"
    length_multiplier: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.total_width_m, self.head_drop_m, self.total_length_m)):
            raise ValueError("Width and head drop must be finite and positive")
        if not all(0 < v < 1 for v in (self.outer_a_fraction, self.inner_fraction)):
            raise ValueError("Split fractions must be strictly between zero and one")
        if self.internal_location not in {"upstream", "central", "downstream"}:
            raise ValueError("Unknown internal location")
        if not math.isfinite(self.length_multiplier) or self.length_multiplier < 1:
            raise ValueError("Length multiplier must be finite and at least one")


def build_original(family, design=None):
    design = design or Design()
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    L = design.total_length_m
    p = Params(L=L, W_total=design.total_width_m, xs=L*.25, xe=L*.75,
               jump=L*.05, max_breaks=1, min_width=0.0)
    internal_start = {"upstream": .35, "central": .4, "downstream": .45}[design.internal_location]*L
    internal_end = internal_start+.2*L
    net = RiverNetworkNX(p)
    a = p.W_total * design.outer_a_fraction
    if family == "centerline":
        net.G.add_edge((p.xs, "main"), (p.xe, "main"), key="center",
                       kind="main", branch="main", w=p.W_total, curve=0)
    else:
        net.instantiate_corridor(a, p.W_total-a)
        if family == "nested":
            net.add_loop("A", internal_start, internal_end, a*design.inner_fraction,
                         a*(1-design.inner_fraction))
        elif family == "crossover":
            net.add_cross("A", "B", internal_start, internal_end, a*design.inner_fraction)
    factors = {}
    for u,v,k,d in net.G.edges(keys=True, data=True):
        target = ((family == "simple" and d.get("branch") == "A") or
                  (family == "nested" and d.get("kind") == "loop" and d.get("curve", 0) > 0) or
                  (family == "crossover" and d.get("kind") == "cross"))
        if target:
            factors[(u,v,k)] = design.length_multiplier
    apply_heads_and_lengths(net, design.head_drop_m, factors)
    return net


def apply_heads_and_lengths(net, head_drop_m, length_factors=None):
    """Assign a head potential, then derive slopes; never multiply slope alone."""
    if not math.isfinite(head_drop_m) or head_drop_m <= 0:
        raise ValueError("Head drop must be finite and positive")
    factors = length_factors or {}
    unknown = set(factors) - set(net.G.edges(keys=True))
    if unknown:
        raise ValueError(f"Unknown edge keys: {unknown}")
    if any(not math.isfinite(v) or v < 1 for v in factors.values()):
        raise ValueError("Length factors must be finite and at least one")
    for _, data in net.G.nodes(data=True):
        data["head_m"] = head_drop_m * (1-data["x"]/net.p.L)
    for u, v, k, data in net.G.edges(keys=True, data=True):
        span = net.G.nodes[v]["x"]-net.G.nodes[u]["x"]
        data["length_base_m"] = span
        data["length_factor"] = factors.get((u,v,k), 1.0)
        data["length_m"] = span * data["length_factor"]
        data["head_drop_m"] = net.G.nodes[u]["head_m"]-net.G.nodes[v]["head_m"]
        data["slope"] = data["head_drop_m"]/data["length_m"]


def planned_states(family, experiment_id):
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    specs = [("original", None, "original", [])]
    if family == "simple":
        specs += [("full", "original", "unit", ["outer"])]
    elif family == "nested":
        specs += [("inner_only", "original", "unit", ["inner"]),
                  ("main_island_only", "original", "island", ["main_island"]),
                  ("full", "original", "unit", ["outer"])]
    elif family == "crossover":
        specs += [("upstream_only", "original", "unit", ["upstream"]),
                  ("downstream_only", "original", "unit", ["downstream"]),
                  ("full", "original", "compound", ["upstream", "downstream"])]
    return [dict(experiment_id=experiment_id, family=family, state_id=s,
                 parent_state_id=p, base_state_id="original", selection_type=t,
                 selected_unit_labels=units, status="planned")
            for s,p,t,units in specs]
