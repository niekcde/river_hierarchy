from __future__ import annotations

"""Final read-only global and local slope audit for candidate contexts.

The audit deliberately uses three complementary comparisons:

* network-global: a link against its fitted end-to-end profile slope;
* domain-local: a solved simple unit/bubble link against that domain's fixed
  boundary-head reference slope;
* immediate topology-local: a link against the median slope of links sharing
  either endpoint.

Only the domain-local comparison is a conformity requirement. Network and
topology contrasts are review signals: genuine waterfalls, different braided
paths, and channel junctions can legitimately differ from neighbouring links.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INSPECTION_SCHEMA_VERSION = "slope-refit-final-inspection-v2"
DOMAIN_FACTOR_REVIEW = 5.0
NETWORK_FACTOR_REVIEW = 50.0
NEIGHBOUR_FACTOR_REVIEW = 20.0
# The candidate solver uses an explicit physical lower bound on every solved
# domain link.  Treat links that intentionally land on that bound separately
# from unbounded local-slope contrasts: the former are an auditable constraint
# activation, not a numerical near-zero artefact.
DOMAIN_FLOOR_RELATIVE_TOLERANCE = 1e-7
DOMAIN_FLOOR_ABSOLUTE_TOLERANCE_M_PER_M = 1e-12


def _canonical_id(value: object) -> str:
    return f"{int(float(value)):02d}"


def _symmetric_factor(value: pd.Series, reference: pd.Series) -> pd.Series:
    ratio = value / reference
    return pd.concat([ratio, 1.0 / ratio], axis=1).max(axis=1)


def _node_neighbour_median(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return median incident-neighbour slope and neighbour count per link."""
    slopes = frame["candidate_slope_m_per_m"].to_numpy(float)
    incident: dict[int, list[int]] = {}
    for position, row in enumerate(frame[["id_us_node", "id_ds_node"]].itertuples(index=False)):
        for node in (int(row.id_us_node), int(row.id_ds_node)):
            incident.setdefault(node, []).append(position)
    medians: list[float] = []
    counts: list[int] = []
    for position, row in enumerate(frame[["id_us_node", "id_ds_node"]].itertuples(index=False)):
        neighbours = set(incident[int(row.id_us_node)]) | set(incident[int(row.id_ds_node)])
        neighbours.discard(position)
        values = slopes[sorted(neighbours)] if neighbours else np.asarray([], dtype=float)
        values = values[np.isfinite(values) & (values > 0.0)]
        medians.append(float(np.median(values)) if len(values) else np.nan)
        counts.append(int(len(values)))
    return pd.Series(medians, index=frame.index), pd.Series(counts, index=frame.index)


def _inspect_example(item: object) -> pd.DataFrame:
    example_id = _canonical_id(item.example_id)
    profile_dir = Path(item.profile_dir)
    candidate_dir = Path(item.candidate_context_dir)
    profile_audit = json.loads((profile_dir / "reference_profile_audit.json").read_text(encoding="utf-8"))
    global_slope = float(profile_audit["summary"]["fitted_global_slope_m_per_m"])
    links = pd.read_parquet(candidate_dir / "candidate_link_hydraulic_context.parquet").copy()
    domains = pd.read_csv(candidate_dir / "candidate_domain_head_loss_audit.csv")
    required = {
        "id_link", "id_us_node", "id_ds_node", "candidate_slope_m_per_m",
        "candidate_domain_kind", "candidate_domain_id", "candidate_slope_source_method",
    }
    missing = sorted(required - set(links.columns))
    if missing:
        raise ValueError(f"Example {example_id} candidate links lack: {missing}")
    links["example_id"] = example_id
    for column in ("id_link", "id_us_node", "id_ds_node", "candidate_domain_id"):
        links[column] = pd.to_numeric(links[column], errors="coerce").astype("Int64")
    links["candidate_slope_m_per_m"] = pd.to_numeric(links["candidate_slope_m_per_m"], errors="coerce")
    links["network_global_reference_slope_m_per_m"] = global_slope
    links["network_global_slope_factor"] = _symmetric_factor(
        links["candidate_slope_m_per_m"], links["network_global_reference_slope_m_per_m"]
    )

    domain_fields = [
        "domain_kind", "domain_id", "status", "local_reference_slope_m_per_m",
        "minimum_domain_slope_m_per_m", "minimum_reference_slope_fraction",
        "max_abs_path_residual_m", "n_paths",
    ]
    available = [field for field in domain_fields if field in domains.columns]
    domain = domains[available].copy()
    domain["domain_id"] = pd.to_numeric(domain["domain_id"], errors="coerce").astype("Int64")
    domain = domain.rename(columns={"domain_kind": "candidate_domain_kind", "domain_id": "candidate_domain_id"})
    links = links.merge(
        domain,
        on=["candidate_domain_kind", "candidate_domain_id"],
        how="left",
        validate="m:1",
    )
    # Path counts can exceed numeric parquet precision and are read from CSV as
    # a mixture of integers and decimal-looking strings. They are provenance,
    # not a calculation input, so retain their exact textual representation.
    if "n_paths" in links.columns:
        links["n_paths"] = links["n_paths"].astype("string")
    links["local_reference_slope_m_per_m"] = pd.to_numeric(
        links.get("local_reference_slope_m_per_m"), errors="coerce"
    )
    links["domain_local_slope_factor"] = _symmetric_factor(
        links["candidate_slope_m_per_m"], links["local_reference_slope_m_per_m"]
    )
    links["minimum_domain_slope_m_per_m"] = pd.to_numeric(
        links.get("minimum_domain_slope_m_per_m"), errors="coerce"
    )
    neighbour_median, neighbour_count = _node_neighbour_median(links)
    links["incident_neighbour_median_slope_m_per_m"] = neighbour_median
    links["n_incident_neighbour_links"] = neighbour_count
    links["topology_neighbour_slope_factor"] = _symmetric_factor(
        links["candidate_slope_m_per_m"], neighbour_median
    )

    solved_domain = links.get("status", pd.Series(index=links.index, dtype=object)).eq("solved")
    links["domain_physical_floor_bound_active"] = (
        solved_domain
        & links["minimum_domain_slope_m_per_m"].notna()
        & np.isclose(
            links["candidate_slope_m_per_m"],
            links["minimum_domain_slope_m_per_m"],
            rtol=DOMAIN_FLOOR_RELATIVE_TOLERANCE,
            atol=DOMAIN_FLOOR_ABSOLUTE_TOLERANCE_M_PER_M,
        )
    )
    links["domain_minimum_slope_satisfied"] = (
        ~solved_domain
        | links["minimum_domain_slope_m_per_m"].isna()
        | links["candidate_slope_m_per_m"].ge(
            links["minimum_domain_slope_m_per_m"] - DOMAIN_FLOOR_ABSOLUTE_TOLERANCE_M_PER_M
        )
    )
    links["domain_local_review"] = solved_domain & links["domain_local_slope_factor"].gt(DOMAIN_FACTOR_REVIEW)
    links["domain_nonfloor_local_review"] = (
        links["domain_local_review"] & ~links["domain_physical_floor_bound_active"]
    )
    links["network_global_review"] = links["network_global_slope_factor"].gt(NETWORK_FACTOR_REVIEW)
    links["topology_neighbour_review"] = (
        links["n_incident_neighbour_links"].gt(0)
        & links["topology_neighbour_slope_factor"].gt(NEIGHBOUR_FACTOR_REVIEW)
    )
    links["review_reason"] = np.select(
        [
            links["domain_physical_floor_bound_active"],
            links["domain_nonfloor_local_review"],
            ~links["domain_local_review"] & links["network_global_review"],
            ~links["domain_local_review"] & ~links["network_global_review"] & links["topology_neighbour_review"],
        ],
        [
            "domain_physical_floor_bound_active",
            "domain_nonfloor_local_factor_over_5",
            "network_global_factor_over_50",
            "topology_neighbour_factor_over_20",
        ],
        default="",
    )
    return links


def inspect_candidate_batch(candidate_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(candidate_root).expanduser().resolve()
    manifest = pd.read_csv(root / "slope_refit_candidate_batch_manifest.csv")
    frames = []
    for item in manifest.itertuples(index=False):
        if str(item.status) not in {"complete", "already_complete"}:
            raise ValueError(f"Candidate batch is incomplete: Example {item.example_id} status={item.status}")
        frames.append(_inspect_example(item))
    links = pd.concat(frames, ignore_index=True)
    summary = links.groupby("example_id", as_index=False).agg(
        n_links=("id_link", "size"),
        min_slope_m_per_m=("candidate_slope_m_per_m", "min"),
        max_slope_m_per_m=("candidate_slope_m_per_m", "max"),
        max_network_global_slope_factor=("network_global_slope_factor", "max"),
        max_domain_local_slope_factor=("domain_local_slope_factor", "max"),
        max_topology_neighbour_slope_factor=("topology_neighbour_slope_factor", "max"),
        n_domain_local_review=("domain_local_review", "sum"),
        n_domain_physical_floor_bound_active=("domain_physical_floor_bound_active", "sum"),
        n_domain_nonfloor_local_review=("domain_nonfloor_local_review", "sum"),
        n_domain_minimum_slope_violations=("domain_minimum_slope_satisfied", lambda values: int((~values).sum())),
        n_network_global_review=("network_global_review", "sum"),
        n_topology_neighbour_review=("topology_neighbour_review", "sum"),
    )
    return links, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect global and local slope outliers in a complete candidate batch.")
    parser.add_argument("candidate_root")
    parser.add_argument("output_dir", help="New empty directory for audit outputs.")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Inspection output must be empty: {output}")
    links, summary = inspect_candidate_batch(args.candidate_root)
    output.mkdir(parents=True, exist_ok=False)
    links.to_parquet(output / "slope_final_link_audit.parquet", index=False)
    summary.to_csv(output / "slope_final_network_summary.csv", index=False)
    review = links.loc[links["review_reason"].ne("")].copy()
    review.to_csv(output / "slope_final_review_queue.csv", index=False)
    payload = {
        "schema_version": INSPECTION_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(Path(args.candidate_root).expanduser().resolve()),
        "thresholds": {
            "domain_local_factor": DOMAIN_FACTOR_REVIEW,
            "network_global_factor": NETWORK_FACTOR_REVIEW,
            "topology_neighbour_factor": NEIGHBOUR_FACTOR_REVIEW,
            "domain_floor_relative_tolerance": DOMAIN_FLOOR_RELATIVE_TOLERANCE,
            "domain_floor_absolute_tolerance_m_per_m": DOMAIN_FLOOR_ABSOLUTE_TOLERANCE_M_PER_M,
        },
        "n_examples": int(summary["example_id"].nunique()),
        "n_links": int(len(links)),
        "n_domain_local_review": int(links["domain_local_review"].sum()),
        "n_domain_physical_floor_bound_active": int(links["domain_physical_floor_bound_active"].sum()),
        "n_domain_nonfloor_local_review": int(links["domain_nonfloor_local_review"].sum()),
        "n_domain_minimum_slope_violations": int((~links["domain_minimum_slope_satisfied"]).sum()),
        "n_network_global_review": int(links["network_global_review"].sum()),
        "n_topology_neighbour_review": int(links["topology_neighbour_review"].sum()),
        "n_unique_review_links": int(len(review)),
    }
    (output / "slope_final_inspection_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
