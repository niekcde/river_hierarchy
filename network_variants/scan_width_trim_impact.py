from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from network_variants.routing_width_application import apply_routing_width_model
from network_variants.width_samples import read_width_samples
from network_variants.width_trim_selection import apply_authoritative_width_trim_selection


def _one(directory: Path, pattern: str) -> Path:
    matches = sorted(path for path in directory.glob(pattern) if not path.name.startswith("._"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {pattern!r} in {directory}; found {len(matches)}.")
    return matches[0]


def _sample_support_metrics(samples: pd.DataFrame, selected_field: str) -> dict[str, float | int | str]:
    ordered = samples.sort_values("sample_index", kind="mergesort")
    x = pd.to_numeric(ordered["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(ordered["y"], errors="coerce").to_numpy(float)
    steps = np.hypot(np.diff(x), np.diff(y))
    finite_steps = steps[np.isfinite(steps) & (steps > 0)]
    cumulative = np.insert(np.cumsum(np.where(np.isfinite(steps), steps, 0.0)), 0, 0.0)
    selected = ordered[selected_field].fillna(False).astype(bool).to_numpy()
    selected_positions = np.flatnonzero(selected)
    selected_count = int(len(selected_positions))
    selected_span = (
        float(cumulative[selected_positions[-1]] - cumulative[selected_positions[0]])
        if selected_count >= 2
        else 0.0
    )
    if selected_count <= 1:
        support_class = "zero_span_single_sample"
    elif selected_count <= 3:
        support_class = "short_2_to_3_samples"
    elif selected_count <= 5:
        support_class = "limited_4_to_5_samples"
    else:
        support_class = "supported_6_plus_samples"
    return {
        "total_sample_count": int(len(ordered)),
        "selected_sample_count": selected_count,
        "median_sample_spacing_m": float(np.median(finite_steps)) if finite_steps.size else np.nan,
        "sampled_link_length_m": float(cumulative[-1]) if len(cumulative) else 0.0,
        "selected_support_length_m": selected_span,
        "selected_support_fraction": (
            selected_span / float(cumulative[-1]) if len(cumulative) and cumulative[-1] > 0 else np.nan
        ),
        "trim_support_class": support_class,
    }


def _node_continuity(
    links: pd.DataFrame,
    *,
    incoming_width_field: str,
    outgoing_width_field: str,
) -> dict[int, tuple[float, float]]:
    frame = links.copy()
    for column in ("id_us_node", "id_ds_node"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    incoming_width = pd.to_numeric(frame[incoming_width_field], errors="coerce")
    outgoing_width = pd.to_numeric(frame[outgoing_width_field], errors="coerce")
    incoming = incoming_width.groupby(frame["id_ds_node"]).sum(min_count=1)
    outgoing = outgoing_width.groupby(frame["id_us_node"]).sum(min_count=1)
    nodes = sorted(set(incoming.dropna().index).intersection(outgoing.dropna().index))
    result: dict[int, tuple[float, float]] = {}
    for node in nodes:
        incoming_sum = float(incoming.loc[node])
        outgoing_sum = float(outgoing.loc[node])
        ratio = outgoing_sum / incoming_sum if incoming_sum > 0 else np.nan
        result[int(node)] = (
            ratio,
            abs(ratio - 1.0) if np.isfinite(ratio) else np.nan,
        )
    return result


def scan_base_width_trim_impact(experiment: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    experiment = Path(experiment)
    base = experiment / "states" / "S000_base" / "variant"
    widths = base / "widths"
    links = gpd.read_file(widths / "links_with_width_families.gpkg")
    samples = read_width_samples(widths, low_memory=False)
    old_routing_audit = pd.read_csv(widths / "routing_width_application.csv", low_memory=False)
    nodes = gpd.read_file(_one(base / "directed", "*directed_nodes.gpkg"))
    with rasterio.open(_one(base / "mask", "*collapsed.tif")) as raster:
        pixel_size_m = float(max(abs(raster.transform.a), abs(raster.transform.e)))
    selected, marked, trim_audit = apply_authoritative_width_trim_selection(links, samples)
    proposed, routing_audit, _ = apply_routing_width_model(
        selected,
        marked,
        nodes,
        example_id=str(links.get("example_id", pd.Series([experiment.name[:2]])).iloc[0]),
        variant_id="S000_base",
        apply_known_base_exclusions=True,
    )
    old = links.set_index(links["id_link"].astype(int), drop=False)
    new = proposed.set_index(proposed["id_link"].astype(int), drop=False)
    route = routing_audit.set_index(routing_audit["id_link"].astype(int), drop=False)
    old_route_audit = old_routing_audit.set_index(
        pd.to_numeric(old_routing_audit["id_link"], errors="raise").astype(int),
        drop=False,
    )
    old_context_continuity = _node_continuity(
        old_routing_audit,
        incoming_width_field="routing_context_width_ds",
        outgoing_width_field="routing_context_width_us",
    )
    new_context_continuity = _node_continuity(
        routing_audit,
        incoming_width_field="routing_context_width_ds",
        outgoing_width_field="routing_context_width_us",
    )
    old_final_continuity = _node_continuity(
        old_routing_audit,
        incoming_width_field="wid_adj_routing",
        outgoing_width_field="wid_adj_routing",
    )
    new_final_continuity = _node_continuity(
        routing_audit,
        incoming_width_field="wid_adj_routing",
        outgoing_width_field="wid_adj_routing",
    )
    trim = trim_audit.set_index(trim_audit["id_link"].astype(int), drop=False)
    sample_groups = {
        int(link_id): group.sort_values("sample_index", kind="mergesort").copy()
        for link_id, group in marked.groupby("id_link", sort=False)
    }
    support_by_link = {
        link_id: _sample_support_metrics(group, "used_in_selected_width")
        for link_id, group in sample_groups.items()
    }
    rows: list[dict[str, object]] = []
    for link_id in sorted(new.index):
        old_wet = pd.to_numeric(pd.Series([old.at[link_id, "wid_adj_wet"]]), errors="coerce").iloc[0]
        new_wet = float(new.at[link_id, "wid_adj_wet"])
        old_routing = pd.to_numeric(
            pd.Series([old.at[link_id, "wid_adj_routing"] if "wid_adj_routing" in old else old_wet]),
            errors="coerce",
        ).iloc[0]
        new_routing = float(new.at[link_id, "wid_adj_routing"])
        us_node = int(new.at[link_id, "id_us_node"])
        ds_node = int(new.at[link_id, "id_ds_node"])

        def continuity_values(
            continuity: dict[int, tuple[float, float]], node_id: int
        ) -> tuple[float, float]:
            return continuity.get(node_id, (np.nan, np.nan))

        old_context_us = continuity_values(old_context_continuity, us_node)
        new_context_us = continuity_values(new_context_continuity, us_node)
        old_context_ds = continuity_values(old_context_continuity, ds_node)
        new_context_ds = continuity_values(new_context_continuity, ds_node)
        old_final_us = continuity_values(old_final_continuity, us_node)
        new_final_us = continuity_values(new_final_continuity, us_node)
        old_final_ds = continuity_values(old_final_continuity, ds_node)
        new_final_ds = continuity_values(new_final_continuity, ds_node)
        support = support_by_link[int(link_id)]
        candidate_group = sample_groups[int(link_id)].copy()
        candidate_start = int(trim.at[link_id, "width_trim_native_candidate_start"])
        candidate_stop = int(trim.at[link_id, "width_trim_native_candidate_stop_exclusive"])
        candidate_group["_native_candidate"] = False
        if 0 <= candidate_start < candidate_stop <= len(candidate_group):
            candidate_group.iloc[
                candidate_start:candidate_stop,
                candidate_group.columns.get_loc("_native_candidate"),
            ] = True
        candidate_support = _sample_support_metrics(candidate_group, "_native_candidate")
        candidate_wet_values = pd.to_numeric(
            candidate_group.loc[candidate_group["_native_candidate"], "width_wet"],
            errors="coerce",
        )
        candidate_wet_values = candidate_wet_values[
            np.isfinite(candidate_wet_values) & candidate_wet_values.gt(0)
        ]
        candidate_wet = (
            float(candidate_wet_values.mean()) if not candidate_wet_values.empty else np.nan
        )
        full_wet_values = pd.to_numeric(candidate_group["width_wet"], errors="coerce")
        full_wet_values = full_wet_values[np.isfinite(full_wet_values) & full_wet_values.gt(0)]

        def distribution(values: pd.Series) -> tuple[float, float, float, float]:
            if values.empty:
                return np.nan, np.nan, np.nan, np.nan
            return (
                float(values.quantile(0.05)),
                float(values.median()),
                float(values.mean()),
                float(values.quantile(0.95)),
            )

        full_distribution = distribution(full_wet_values)
        candidate_distribution = distribution(candidate_wet_values)
        wet_relative_change = (new_wet / float(old_wet) - 1.0) if old_wet > 0 else np.nan
        candidate_wet_relative_change = (
            candidate_wet / float(old_wet) - 1.0
            if old_wet > 0 and np.isfinite(candidate_wet)
            else np.nan
        )
        native_fallback = trim.at[link_id, "width_trim_method"] == "native_rivgraph_fallback"
        rejected_short_fallback = "native_endpoint_trim_insufficient_support" in str(
            trim.at[link_id, "width_trim_fallback_reason"]
        )
        is_native_candidate = native_fallback or rejected_short_fallback
        if is_native_candidate and (
            int(candidate_support["selected_sample_count"]) <= 3
            or (
                np.isfinite(candidate_wet_relative_change)
                and abs(candidate_wet_relative_change) >= 0.25
            )
        ):
            review_priority = "mandatory"
        elif is_native_candidate and (
            int(candidate_support["selected_sample_count"]) <= 5
            or (
                np.isfinite(candidate_wet_relative_change)
                and abs(candidate_wet_relative_change) >= 0.10
            )
        ):
            review_priority = "recommended"
        elif is_native_candidate:
            review_priority = "routine"
        else:
            review_priority = "not_native_fallback"
        rows.append(
            {
                "experiment": experiment.name,
                "id_link": int(link_id),
                "width_trim_method": trim.at[link_id, "width_trim_method"],
                "width_trim_fallback_reason": trim.at[link_id, "width_trim_fallback_reason"],
                "native_candidate_status": (
                    "accepted" if native_fallback
                    else "rejected_insufficient_support" if rejected_short_fallback
                    else "not_available"
                ),
                "trim_start": int(trim.at[link_id, "width_trim_start_sample_index"]),
                "trim_stop_exclusive": int(trim.at[link_id, "width_trim_stop_sample_index_exclusive"]),
                **support,
                "pixel_size_m": pixel_size_m,
                "selected_support_pixels": (
                    float(support["selected_support_length_m"]) / pixel_size_m
                    if pixel_size_m > 0 else np.nan
                ),
                "native_candidate_sample_count": int(candidate_support["selected_sample_count"]),
                "native_candidate_support_length_m": float(
                    candidate_support["selected_support_length_m"]
                ),
                "native_candidate_support_pixels": (
                    float(candidate_support["selected_support_length_m"]) / pixel_size_m
                    if pixel_size_m > 0 else np.nan
                ),
                "native_candidate_support_fraction": float(
                    candidate_support["selected_support_fraction"]
                ),
                "native_candidate_support_class": candidate_support["trim_support_class"],
                "native_candidate_wid_adj_wet": candidate_wet,
                "native_candidate_wet_relative_change": candidate_wet_relative_change,
                "full_wet_p05": full_distribution[0],
                "full_wet_median": full_distribution[1],
                "full_wet_mean": full_distribution[2],
                "full_wet_p95": full_distribution[3],
                "native_candidate_wet_p05": candidate_distribution[0],
                "native_candidate_wet_median": candidate_distribution[1],
                "native_candidate_wet_mean": candidate_distribution[2],
                "native_candidate_wet_p95": candidate_distribution[3],
                "review_priority": review_priority,
                "old_wid_adj_wet": float(old_wet),
                "new_wid_adj_wet": new_wet,
                "wet_absolute_change": new_wet - float(old_wet),
                "wet_relative_change": wet_relative_change,
                "old_wid_adj_routing": float(old_routing),
                "new_wid_adj_routing": new_routing,
                "routing_absolute_change": new_routing - float(old_routing),
                "routing_relative_change": (
                    new_routing / float(old_routing) - 1.0 if old_routing > 0 else np.nan
                ),
                "new_context_us": float(route.at[link_id, "routing_context_width_us"]),
                "new_context_ds": float(route.at[link_id, "routing_context_width_ds"]),
                "old_width_model_prediction": old_route_audit.at[
                    link_id, "width_model_prediction"
                ],
                "new_width_model_prediction": route.at[link_id, "width_model_prediction"],
                "old_width_qc_decision": old_route_audit.at[link_id, "width_qc_decision"],
                "new_width_qc_decision": route.at[link_id, "width_qc_decision"],
                "old_width_qc_reason": old_route_audit.at[link_id, "width_qc_reason"],
                "new_width_qc_reason": route.at[link_id, "width_qc_reason"],
                "old_width_correction_scale_factor": old_route_audit.at[
                    link_id, "width_correction_scale_factor"
                ],
                "new_width_correction_scale_factor": route.at[
                    link_id, "width_correction_scale_factor"
                ],
                "old_context_continuity_ratio_us_node": old_context_us[0],
                "new_context_continuity_ratio_us_node": new_context_us[0],
                "old_context_continuity_abs_deviation_us_node": old_context_us[1],
                "new_context_continuity_abs_deviation_us_node": new_context_us[1],
                "old_context_continuity_ratio_ds_node": old_context_ds[0],
                "new_context_continuity_ratio_ds_node": new_context_ds[0],
                "old_context_continuity_abs_deviation_ds_node": old_context_ds[1],
                "new_context_continuity_abs_deviation_ds_node": new_context_ds[1],
                "old_final_continuity_ratio_us_node": old_final_us[0],
                "new_final_continuity_ratio_us_node": new_final_us[0],
                "old_final_continuity_abs_deviation_us_node": old_final_us[1],
                "new_final_continuity_abs_deviation_us_node": new_final_us[1],
                "old_final_continuity_ratio_ds_node": old_final_ds[0],
                "new_final_continuity_ratio_ds_node": new_final_ds[0],
                "old_final_continuity_abs_deviation_ds_node": old_final_ds[1],
                "new_final_continuity_abs_deviation_ds_node": new_final_ds[1],
            }
        )
    frame = pd.DataFrame.from_records(rows)
    frame["wet_change_material"] = ~np.isclose(
        frame["old_wid_adj_wet"], frame["new_wid_adj_wet"], rtol=1e-6, atol=1e-6
    )
    frame["routing_change_material"] = ~np.isclose(
        frame["old_wid_adj_routing"], frame["new_wid_adj_routing"], rtol=1e-6, atol=1e-6
    )
    changed = frame.loc[frame["wet_change_material"] | frame["routing_change_material"]]
    summary = {
        "experiment": experiment.name,
        "link_count": int(len(frame)),
        "native_fallback_count": int(frame["width_trim_method"].eq("native_rivgraph_fallback").sum()),
        "rejected_short_native_fallback_count": int(
            frame["native_candidate_status"].eq("rejected_insufficient_support").sum()
        ),
        "mandatory_native_fallback_review_count": int(frame["review_priority"].eq("mandatory").sum()),
        "recommended_native_fallback_review_count": int(frame["review_priority"].eq("recommended").sum()),
        "single_sample_native_fallback_count": int(
            (
                frame["native_candidate_status"].isin(
                    ["accepted", "rejected_insufficient_support"]
                )
                & frame["native_candidate_sample_count"].eq(1)
            ).sum()
        ),
        "no_valid_core_count": int(frame["width_trim_method"].eq("untrimmed_no_valid_core").sum()),
        "changed_wet_link_count": int(frame["wet_change_material"].sum()),
        "changed_routing_link_count": int(frame["routing_change_material"].sum()),
        "changed_link_ids": sorted(changed["id_link"].astype(int).tolist()),
        "max_wet_relative_change": float(frame["wet_relative_change"].abs().max()),
        "max_routing_relative_change": float(frame["routing_relative_change"].abs().max()),
    }
    return frame, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only inventory of authoritative trim-selection impact.")
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="Optional combined CSV output; no experiment files are changed.")
    args = parser.parse_args(argv)
    frames: list[pd.DataFrame] = []
    for experiment in args.experiments:
        frame, summary = scan_base_width_trim_impact(experiment)
        frames.append(frame)
        print(f"\n{summary['experiment']}")
        for key, value in summary.items():
            if key != "experiment":
                print(f"  {key}: {value}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nWrote read-only impact report to {args.output}")
        review_output = args.output.with_name(
            f"{args.output.stem}_native_fallback_review{args.output.suffix or '.csv'}"
        )
        review = combined.loc[
            combined["native_candidate_status"].isin(
                ["accepted", "rejected_insufficient_support"]
            )
        ].sort_values(
            ["review_priority", "experiment", "id_link"],
            key=lambda values: (
                values.map({"mandatory": 0, "recommended": 1, "routine": 2})
                if values.name == "review_priority"
                else values
            ),
            kind="mergesort",
        )
        review.to_csv(review_output, index=False)
        print(f"Wrote native-fallback review queue to {review_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
