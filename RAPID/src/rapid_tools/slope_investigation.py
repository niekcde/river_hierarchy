from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "network_variants" / "output"
REPO_ROOT = Path(__file__).resolve().parents[3]

DISTANCE_LABELS = {
    "node_network_dist_out_m": "Node downstream network distance [m]",
    "node_reference_section_dist_out_m": "Node downstream reference-section distance [m]",
    "node_sword_dist_out": "Node downstream SWORD distance [m]",
}

WSE_LABELS = {
    "node_wse_raw": "Observed / requested node WSE",
    "sword_wse_raw": "Pure SWORD node WSE",
    "rapid_wse_for_slope": "Monotone WSE used for slope",
}


def _normalize_example_id(value: str | int) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Example IDs cannot be empty.")
    return text.zfill(2) if text.isdigit() and len(text) < 2 else text


def _normalize_state_ids(
    state_ids: Sequence[str] | Mapping[str, Sequence[str]] | None,
    example_ids: Sequence[str],
) -> dict[str, list[str]] | None:
    if state_ids is None:
        return None
    if isinstance(state_ids, Mapping):
        return {
            _normalize_example_id(example_id): [str(state_id) for state_id in values]
            for example_id, values in state_ids.items()
        }
    normalized = [str(state_id) for state_id in state_ids]
    return {_normalize_example_id(example_id): list(normalized) for example_id in example_ids}


def _experiment_dir(
    example_id: str | int,
    *,
    mode: str = "indep",
    anchored_suffix: str = "__anchored",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    normalized = _normalize_example_id(example_id)
    mode_name = "compound_bubbles" if str(mode).strip().lower() == "compound" else "indep"
    return Path(output_root) / f"{normalized}_{mode_name}{anchored_suffix}"


def _state_dir(selection_row: pd.Series) -> Path:
    return Path(selection_row["experiment_dir"]) / "states" / str(selection_row["state_id"])


def _pick_distance_column(columns: Sequence[str]) -> str:
    for candidate in ("node_reference_section_dist_out_m", "node_network_dist_out_m", "node_sword_dist_out"):
        if candidate in columns:
            return candidate
    raise ValueError("No supported distance column found in node hydraulic-context table.")


def list_slope_states(
    example_ids: Sequence[str | int],
    *,
    mode: str = "indep",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    anchored_suffix: str = "__anchored",
    state_ids: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    status: str | None = "failed",
    include_base: bool = True,
    limit_per_example: int | None = None,
) -> pd.DataFrame:
    normalized_examples = [_normalize_example_id(example_id) for example_id in example_ids]
    normalized_state_ids = _normalize_state_ids(state_ids, normalized_examples)

    rows: list[dict[str, Any]] = []
    for example_id in normalized_examples:
        exp_dir = _experiment_dir(
            example_id,
            mode=mode,
            anchored_suffix=anchored_suffix,
            output_root=output_root,
        )
        if not exp_dir.exists():
            continue

        registry_path = exp_dir / "rapid_prep_registry.csv"
        registry = pd.read_csv(registry_path) if registry_path.exists() and registry_path.stat().st_size > 1 else pd.DataFrame()
        if not registry.empty and "status" not in registry.columns:
            registry["status"] = pd.NA

        wanted_state_ids = None if normalized_state_ids is None else normalized_state_ids.get(example_id, [])
        if wanted_state_ids is not None:
            wanted = pd.Index(wanted_state_ids)
            selection = registry.loc[registry["state_id"].astype(str).isin(wanted)].copy() if not registry.empty else pd.DataFrame({"state_id": wanted})
        else:
            selection = registry.copy()
            if status is not None and not selection.empty and "status" in selection.columns:
                selection = selection.loc[selection["status"].astype(str).eq(str(status))]
            if limit_per_example is not None and limit_per_example > 0 and not selection.empty:
                selection = selection.head(int(limit_per_example))

        if include_base:
            base_present = not selection.empty and selection["state_id"].astype(str).eq("S000_base").any()
            if not base_present:
                if not registry.empty and registry["state_id"].astype(str).eq("S000_base").any():
                    selection = pd.concat(
                        [registry.loc[registry["state_id"].astype(str).eq("S000_base")], selection],
                        ignore_index=True,
                    )
                else:
                    selection = pd.concat([pd.DataFrame([{"state_id": "S000_base"}]), selection], ignore_index=True)

        if selection.empty and include_base:
            selection = pd.DataFrame([{"state_id": "S000_base"}])

        for _, row in selection.iterrows():
            state_id = str(row["state_id"])
            state_dir = exp_dir / "states" / state_id
            rows.append(
                {
                    "example_id": example_id,
                    "mode": mode,
                    "experiment_dir": str(exp_dir),
                    "state_id": state_id,
                    "state_dir": str(state_dir),
                    "state_role": row.get("state_role", "base" if state_id == "S000_base" else "derived"),
                    "prep_status": row.get("status"),
                    "prep_error": row.get("error"),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.drop_duplicates(subset=["example_id", "mode", "state_id"]).reset_index(drop=True)
    result["state_label"] = result["example_id"].astype(str) + " · " + result["state_id"].astype(str)
    return result


def build_slope_node_frame(selection: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in selection.iterrows():
        state_dir = Path(row["state_dir"])
        node_path = state_dir / "hydraulic_context" / "node_hydraulic_context.csv"
        if not node_path.exists() or node_path.stat().st_size <= 1:
            continue
        frame = pd.read_csv(node_path)
        sword_match_path = state_dir / "variant" / "matching" / "node_sword_match.csv"
        if sword_match_path.exists() and sword_match_path.stat().st_size > 1:
            sword = pd.read_csv(sword_match_path)
            sword_cols = [
                c
                for c in [
                    "id_node",
                    "sword_node_id",
                    "sword_reach_id",
                    "sword_source_file",
                    "sword_wse",
                    "sword_dist_out",
                    "sword_wse_field",
                    "sword_wse_fill_method",
                    "sword_match_method",
                    "sword_match_from_parent",
                    "sword_match_within_tolerance",
                ]
                if c in sword.columns
            ]
            if sword_cols:
                rename_map = {
                    "sword_wse": "sword_wse_raw",
                    "sword_dist_out": "sword_dist_out_raw",
                    "sword_wse_field": "sword_wse_field_raw",
                    "sword_wse_fill_method": "sword_wse_fill_method_raw",
                    "sword_match_method": "sword_match_method_raw",
                    "sword_match_from_parent": "sword_match_from_parent_raw",
                    "sword_match_within_tolerance": "sword_match_within_tolerance_raw",
                }
                frame = frame.merge(
                    sword[sword_cols].rename(columns=rename_map),
                    on="id_node",
                    how="left",
                )
        frame["example_id"] = row["example_id"]
        frame["mode"] = row["mode"]
        frame["state_id"] = row["state_id"]
        frame["state_label"] = row["state_label"]
        frame["prep_status"] = row.get("prep_status")
        frame["distance_column_used"] = _pick_distance_column(frame.columns)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    if "id_node" in result.columns:
        result["id_node"] = pd.to_numeric(result["id_node"], errors="coerce").astype("Int64")
    return result


def build_actual_sword_node_frame(
    node_frame: pd.DataFrame,
    *,
    example_id: str | int | None = None,
    state_id: str | None = None,
) -> pd.DataFrame:
    if node_frame.empty:
        return pd.DataFrame()

    frame = node_frame.copy()
    if example_id is not None:
        frame = frame.loc[frame["example_id"].astype(str).eq(_normalize_example_id(example_id))].copy()
    if state_id is not None:
        frame = frame.loc[frame["state_id"].astype(str).eq(str(state_id))].copy()
    if frame.empty:
        return pd.DataFrame()
    required = {"sword_source_file", "sword_reach_id"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for (example, state, state_label), group in frame.groupby(["example_id", "state_id", "state_label"], dropna=False):
        source_files = [Path(p) for p in group["sword_source_file"].dropna().astype(str).unique() if str(p).strip()]
        reach_ids = pd.to_numeric(group["sword_reach_id"], errors="coerce").dropna().astype(int).unique().tolist()
        requested_fields = [str(v).strip() for v in group.get("sword_wse_field_raw", pd.Series(dtype=object)).dropna().astype(str).unique() if str(v).strip()]
        preferred_wse_field = requested_fields[0] if requested_fields else "wse_obs_p50"
        for source_file in source_files:
            if not source_file.exists():
                continue
            columns = ["node_id", "reach_id", "dist_out", preferred_wse_field]
            try:
                sword_nodes = pd.read_parquet(source_file, columns=columns)
            except Exception:
                sword_nodes = pd.read_parquet(source_file)
            if "reach_id" not in sword_nodes.columns or "dist_out" not in sword_nodes.columns:
                continue
            local = sword_nodes.loc[pd.to_numeric(sword_nodes["reach_id"], errors="coerce").isin(reach_ids)].copy()
            if local.empty:
                continue
            if preferred_wse_field not in local.columns:
                continue
            local = local.rename(
                columns={
                    "node_id": "sword_node_id_actual",
                    "reach_id": "sword_reach_id_actual",
                    "dist_out": "sword_dist_out_actual",
                    preferred_wse_field: "sword_wse_actual",
                }
            )
            local["example_id"] = str(example)
            local["state_id"] = str(state)
            local["state_label"] = str(state_label)
            local["sword_wse_field_actual"] = preferred_wse_field
            local["sword_source_file_actual"] = str(source_file)
            rows.append(local)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    for column in ("sword_node_id_actual", "sword_reach_id_actual"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    return result


def build_slope_link_frame(selection: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in selection.iterrows():
        state_dir = Path(row["state_dir"])
        link_context_path = state_dir / "hydraulic_context" / "link_hydraulic_context.csv"
        widths_path = state_dir / "variant" / "widths" / "links_with_width_families.gpkg"
        lineage_path = state_dir / "variant" / "matching" / "link_lineage.csv"
        if not link_context_path.exists() or link_context_path.stat().st_size <= 1 or not widths_path.exists():
            continue

        link_frame = pd.read_csv(link_context_path)
        geo = gpd.read_file(widths_path)
        geo["centroid_x"] = geo.geometry.centroid.x.astype(float)
        geo["centroid_y"] = geo.geometry.centroid.y.astype(float)
        geo_cols = [
            c
            for c in [
                "id_link",
                "id_us_node",
                "id_ds_node",
                "wid_adj_routing",
                "wid_adj_wet",
                "len",
                "centroid_x",
                "centroid_y",
            ]
            if c in geo.columns
        ]
        merged = link_frame.merge(geo[geo_cols], on=["id_link", "id_us_node", "id_ds_node"], how="left")

        if lineage_path.exists() and lineage_path.stat().st_size > 1:
            lineage = pd.read_csv(lineage_path)
            lineage_cols = [
                c
                for c in [
                    "id_link",
                    "dominant_parent_link_id",
                    "matched_parent_link_ids",
                    "lineage_type",
                    "lineage_method",
                    "matched_parent_node_path",
                ]
                if c in lineage.columns
            ]
            if lineage_cols:
                merged = merged.merge(lineage[lineage_cols], on="id_link", how="left")

        merged["example_id"] = row["example_id"]
        merged["mode"] = row["mode"]
        merged["state_id"] = row["state_id"]
        merged["state_label"] = row["state_label"]
        merged["prep_status"] = row.get("prep_status")
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    for column in ("id_link", "id_us_node", "id_ds_node"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    return result


def _load_prep_manifest(experiment_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(experiment_dir) / "rapid_prep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing rapid prep manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def build_low_k_parent_summary(
    selection: pd.DataFrame,
    *,
    state_id: str | None = "S000_base",
    top_n: int = 20,
) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()

    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))

    from .k_values import KValueConfig
    from .prep import (
        _resolve_link_width_samples_path,
        apply_subreach_widths_from_samples,
        compute_k_values,
        split_links_into_subreaches,
    )

    rows: list[pd.DataFrame] = []
    work = selection.copy()
    if state_id is not None:
        work = work.loc[work["state_id"].astype(str).eq(str(state_id))].copy()
    if work.empty:
        return pd.DataFrame()

    for _, row in work.iterrows():
        state_dir = Path(row["state_dir"])
        experiment_dir = Path(row["experiment_dir"])
        manifest = _load_prep_manifest(experiment_dir)
        prep_config = manifest.get("prep_config", {})
        kb_summary = manifest.get("resolved_kb_summary", {})

        widths_path = state_dir / "variant" / "widths" / "links_with_width_families.gpkg"
        nodes_path = next((state_dir / "variant" / "directed").glob("*_directed_nodes.gpkg"))
        hydro_path = state_dir / "hydraulic_context" / "link_hydraulic_context.csv"
        links = gpd.read_file(widths_path)
        nodes = gpd.read_file(nodes_path)
        hydro = pd.read_csv(hydro_path)
        hydro_cols = [c for c in ["id_link", "id_us_node", "id_ds_node", "link_length_m", "slope_used"] if c in hydro.columns]
        prepared = links.merge(hydro[hydro_cols], on=["id_link", "id_us_node", "id_ds_node"], how="left")
        prepared["parent_centroid_x"] = prepared.geometry.centroid.x.astype(float)
        prepared["parent_centroid_y"] = prepared.geometry.centroid.y.astype(float)

        rapid_links, _ = split_links_into_subreaches(
            prepared,
            nodes,
            target_length_m=prep_config.get("target_subreach_length_m"),
        )
        if bool(prep_config.get("vary_width_by_subreach", False)):
            rapid_links = apply_subreach_widths_from_samples(
                rapid_links,
                prepared,
                link_width_samples_path=_resolve_link_width_samples_path(widths_path),
                sample_field=str(prep_config.get("subreach_width_sample_field", "width_wet")),
                fallback_width_field=str(prep_config.get("width_field", "wid_adj_routing")),
            )
            width_field = "rapid_subreach_width_m"
        else:
            width_field = str(prep_config.get("width_field", "wid_adj_routing"))

        rapid_links = compute_k_values(
            rapid_links,
            config=KValueConfig(
                width_field=width_field,
                x_value=float(prep_config.get("x_value", 0.1)),
                kb_value=float(kb_summary.get("kb_value", prep_config.get("kb_value", 20.0))),
                kb_source_method=str(kb_summary.get("kb_source_method", prep_config.get("kb_mode", "fixed"))),
                kb_reference_discharge_cms=kb_summary.get("reference_discharge_cms"),
                kb_reference_width_m=kb_summary.get("reference_width_m"),
                kb_reference_slope=kb_summary.get("reference_slope"),
                kb_reference_depth_m=kb_summary.get("reference_depth_m"),
                n_manning=float(prep_config.get("n_manning", 0.35)),
                min_width=float(prep_config.get("min_width", 1.0)),
                min_effective_length_m=prep_config.get("min_effective_length_for_k_m"),
                use_celerity_capping=bool(prep_config.get("use_celerity_capping", False)),
                min_celerity_mps=float(prep_config.get("min_celerity_mps", 0.05)),
                max_celerity_mps=float(prep_config.get("max_celerity_mps", 2.524)),
            ),
        )
        finite_k = pd.to_numeric(rapid_links["rapid_k"], errors="coerce").dropna().astype(float)
        x_value = float(prep_config.get("x_value", 0.1))
        dt_min = float((2.0 * finite_k * x_value).max()) if not finite_k.empty else np.nan
        dt_max = float((2.0 * finite_k * (1.0 - x_value)).min()) if not finite_k.empty else np.nan

        subreach_summary = (
            rapid_links.groupby("parent_link_id", as_index=False)
            .agg(
                min_rapid_k=("rapid_k", "min"),
                median_rapid_k=("rapid_k", "median"),
                max_rapid_k=("rapid_k", "max"),
                max_rapid_celerity_mps=("rapid_celerity_mps", "max"),
                min_rapid_celerity_mps=("rapid_celerity_mps", "min"),
                min_subreach_width_m=("rapid_subreach_width_m", "min"),
                max_subreach_width_m=("rapid_subreach_width_m", "max"),
                n_subreaches=("reach_id", "size"),
            )
        )
        parent_meta = prepared[
            [
                c
                for c in [
                    "id_link",
                    "id_us_node",
                    "id_ds_node",
                    "link_length_m",
                    "len",
                    "wid_adj_routing",
                    "wid_adj_wet",
                    "slope_used",
                    "parent_centroid_x",
                    "parent_centroid_y",
                ]
                if c in prepared.columns
            ]
        ].drop_duplicates(subset=["id_link"])
        parent_meta = parent_meta.rename(columns={"id_link": "parent_link_id"})
        summary = subreach_summary.merge(parent_meta, on="parent_link_id", how="left")
        summary["example_id"] = row["example_id"]
        summary["mode"] = row["mode"]
        summary["state_id"] = row["state_id"]
        summary["state_label"] = row["state_label"]
        summary["dt_min"] = dt_min
        summary["dt_max"] = dt_max
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["example_id", "state_id", "min_rapid_k", "parent_link_id"]).reset_index(drop=True)
    if top_n is not None and top_n > 0:
        result = (
            result.groupby(["example_id", "state_id"], group_keys=False)
            .head(int(top_n))
            .reset_index(drop=True)
        )
    return result


def build_slope_profile_long_frame(
    node_frame: pd.DataFrame,
    *,
    distance_column: str | None = None,
    value_columns: Sequence[str] = ("node_wse_raw", "rapid_wse_for_slope"),
) -> pd.DataFrame:
    if node_frame.empty:
        return pd.DataFrame()
    if distance_column is None:
        distance_column = _pick_distance_column(node_frame.columns)

    keep_value_columns = [column for column in value_columns if column in node_frame.columns]
    if not keep_value_columns:
        raise ValueError("None of the requested WSE columns are present in node_frame.")

    id_vars = [
        column
        for column in [
            "example_id",
            "mode",
            "state_id",
            "state_label",
            "prep_status",
            "id_node",
            distance_column,
            "node_network_dist_out_m",
            "node_reference_section_dist_out_m",
            "node_sword_dist_out",
            "rapid_wse_source_method",
            "rapid_wse_profile_residual",
        ]
        if column in node_frame.columns
    ]
    id_vars = list(dict.fromkeys(id_vars))
    long_frame = node_frame.melt(
        id_vars=id_vars,
        value_vars=keep_value_columns,
        var_name="wse_kind",
        value_name="wse_value",
    )
    long_frame["wse_kind_label"] = long_frame["wse_kind"].map(WSE_LABELS).fillna(long_frame["wse_kind"])
    long_frame["distance_column_used"] = distance_column
    long_frame = long_frame.dropna(subset=[distance_column, "wse_value"]).copy()
    return long_frame


def identify_slope_seed_links(
    link_frame: pd.DataFrame,
    *,
    min_ratio: float = 3.0,
    allowed_source_method: str = "raw_wse",
) -> pd.DataFrame:
    if link_frame.empty:
        return pd.DataFrame()
    frame = link_frame.copy()
    if "slope_section_ratio" not in frame.columns:
        frame["slope_section_ratio"] = np.nan
    mask = pd.Series(True, index=frame.index)
    if "slope_source_method" in frame.columns:
        mask &= frame["slope_source_method"].astype(str).eq(allowed_source_method)
    if "slope_section_ratio" in frame.columns:
        mask &= pd.to_numeric(frame["slope_section_ratio"], errors="coerce").ge(float(min_ratio))
    result = frame.loc[mask].copy()
    sort_cols = [c for c in ["example_id", "state_id", "slope_section_ratio", "raw_slope"] if c in result.columns]
    ascending = [True, True, False, False][: len(sort_cols)]
    if sort_cols:
        result = result.sort_values(sort_cols, ascending=ascending)
    return result


def plot_slope_profiles(
    node_frame: pd.DataFrame,
    *,
    distance_column: str | None = None,
    value_columns: Sequence[str] = ("node_wse_raw", "rapid_wse_for_slope"),
    facet_by_example: bool = True,
) -> Any:
    long_frame = build_slope_profile_long_frame(
        node_frame,
        distance_column=distance_column,
        value_columns=value_columns,
    )
    if long_frame.empty:
        raise ValueError("No node-profile data available for plotting.")

    distance_column = str(long_frame["distance_column_used"].iloc[0])
    fig = px.line(
        long_frame,
        x=distance_column,
        y="wse_value",
        color="state_label",
        line_dash="wse_kind_label",
        facet_col="example_id" if facet_by_example and long_frame["example_id"].nunique() > 1 else None,
        hover_data=[c for c in ["example_id", "state_id", "id_node", "wse_kind_label", "rapid_wse_source_method"] if c in long_frame.columns],
        labels={
            distance_column: DISTANCE_LABELS.get(distance_column, distance_column),
            "wse_value": "Water-surface elevation",
            "state_label": "State",
            "wse_kind_label": "Profile",
            "example_id": "Example ID",
        },
        title="Node WSE profiles used for slope diagnostics",
    )
    fig.update_traces(mode="lines+markers")
    return fig


def plot_raw_vs_monotone_profile(
    node_frame: pd.DataFrame,
    *,
    example_id: str | int | None = None,
    state_id: str | None = None,
    distance_column: str | None = None,
) -> Any:
    if node_frame.empty:
        raise ValueError("No node-profile data available for plotting.")

    frame = node_frame.copy()
    if example_id is not None:
        frame = frame.loc[frame["example_id"].astype(str).eq(_normalize_example_id(example_id))].copy()
    if state_id is not None:
        frame = frame.loc[frame["state_id"].astype(str).eq(str(state_id))].copy()
    if frame.empty:
        raise ValueError("No node rows remain after filtering by example_id/state_id.")

    long_frame = build_slope_profile_long_frame(
        frame,
        distance_column=distance_column,
        value_columns=("node_wse_raw", "rapid_wse_for_slope"),
    )
    if long_frame.empty:
        raise ValueError("No raw/monotone node-profile data available for plotting.")

    distance_column = str(long_frame["distance_column_used"].iloc[0])
    fig = px.scatter(
        long_frame,
        x=distance_column,
        y="wse_value",
        color="wse_kind_label",
        symbol="wse_kind_label",
        facet_col="state_label" if long_frame["state_label"].nunique() > 1 else None,
        hover_data=[c for c in ["example_id", "state_id", "id_node", "rapid_wse_profile_residual"] if c in long_frame.columns],
        labels={
            distance_column: DISTANCE_LABELS.get(distance_column, distance_column),
            "wse_value": "Water-surface elevation",
            "wse_kind_label": "Profile",
            "state_label": "State",
        },
        title="Raw node WSE versus monotone profile used for slope",
    )
    return fig


def plot_sword_vs_monotone_profile(
    node_frame: pd.DataFrame,
    *,
    example_id: str | int | None = None,
    state_id: str | None = None,
    distance_column: str | None = None,
) -> Any:
    if node_frame.empty:
        raise ValueError("No node-profile data available for plotting.")
    if "sword_wse_raw" not in node_frame.columns:
        raise ValueError("node_frame does not include pure SWORD node values. Rebuild with build_slope_node_frame().")

    frame = node_frame.copy()
    if example_id is not None:
        frame = frame.loc[frame["example_id"].astype(str).eq(_normalize_example_id(example_id))].copy()
    if state_id is not None:
        frame = frame.loc[frame["state_id"].astype(str).eq(str(state_id))].copy()
    if frame.empty:
        raise ValueError("No node rows remain after filtering by example_id/state_id.")

    long_frame = build_slope_profile_long_frame(
        frame,
        distance_column=distance_column or "node_sword_dist_out",
        value_columns=("sword_wse_raw", "rapid_wse_for_slope"),
    )
    if long_frame.empty:
        raise ValueError("No pure-SWORD/monotone node-profile data available for plotting.")

    distance_column = str(long_frame["distance_column_used"].iloc[0])
    fig = px.scatter(
        long_frame,
        x=distance_column,
        y="wse_value",
        color="wse_kind_label",
        symbol="wse_kind_label",
        facet_col="state_label" if long_frame["state_label"].nunique() > 1 else None,
        hover_data=[
            c
            for c in [
                "example_id",
                "state_id",
                "id_node",
                "rapid_wse_profile_residual",
                "sword_wse_fill_method_raw",
                "sword_match_method_raw",
            ]
            if c in long_frame.columns
        ],
        labels={
            distance_column: DISTANCE_LABELS.get(distance_column, distance_column),
            "wse_value": "Water-surface elevation",
            "wse_kind_label": "Profile",
            "state_label": "State",
        },
        title="Pure SWORD node WSE versus monotone profile used for slope",
    )
    return fig


def plot_actual_sword_vs_state_vs_monotone(
    node_frame: pd.DataFrame,
    *,
    example_id: str | int | None = None,
    state_id: str | None = None,
) -> Any:
    if node_frame.empty:
        raise ValueError("No node-profile data available for plotting.")

    frame = node_frame.copy()
    if example_id is not None:
        frame = frame.loc[frame["example_id"].astype(str).eq(_normalize_example_id(example_id))].copy()
    if state_id is not None:
        frame = frame.loc[frame["state_id"].astype(str).eq(str(state_id))].copy()
    if frame.empty:
        raise ValueError("No node rows remain after filtering by example_id/state_id.")

    actual = build_actual_sword_node_frame(frame)
    if actual.empty:
        raise ValueError("No actual SWORD corridor nodes could be loaded for the selected states.")

    actual_long = actual.rename(
        columns={
            "sword_dist_out_actual": "distance_plot",
            "sword_wse_actual": "wse_value",
        }
    ).copy()
    actual_long["profile_kind"] = "Actual SWORD corridor nodes"
    actual_long["plot_source"] = "actual_sword"

    state_cols = [c for c in ["example_id", "state_id", "state_label", "id_node", "sword_dist_out_raw", "sword_wse_raw", "rapid_wse_for_slope"] if c in frame.columns]
    state_long = frame[state_cols].melt(
        id_vars=[c for c in ["example_id", "state_id", "state_label", "id_node", "sword_dist_out_raw"] if c in frame.columns],
        value_vars=[c for c in ["sword_wse_raw", "rapid_wse_for_slope"] if c in frame.columns],
        var_name="profile_raw_kind",
        value_name="wse_value",
    )
    state_long = state_long.rename(columns={"sword_dist_out_raw": "distance_plot"}).dropna(subset=["distance_plot", "wse_value"]).copy()
    state_long["profile_kind"] = state_long["profile_raw_kind"].map(
        {
            "sword_wse_raw": "Matched state-node SWORD values",
            "rapid_wse_for_slope": "Monotone profile used for slope",
        }
    ).fillna(state_long["profile_raw_kind"])
    state_long["plot_source"] = np.where(
        state_long["profile_raw_kind"].eq("rapid_wse_for_slope"),
        "monotone_state",
        "matched_state_sword",
    )

    plot_frame = pd.concat(
        [
            actual_long[["example_id", "state_id", "state_label", "distance_plot", "wse_value", "profile_kind", "plot_source"]],
            state_long[["example_id", "state_id", "state_label", "distance_plot", "wse_value", "profile_kind", "plot_source"]],
        ],
        ignore_index=True,
    )

    fig = px.scatter(
        plot_frame,
        x="distance_plot",
        y="wse_value",
        color="profile_kind",
        symbol="profile_kind",
        facet_col="state_label" if plot_frame["state_label"].nunique() > 1 else None,
        hover_data=[c for c in ["example_id", "state_id", "profile_kind"] if c in plot_frame.columns],
        labels={
            "distance_plot": DISTANCE_LABELS.get("node_sword_dist_out", "Downstream SWORD distance [m]"),
            "wse_value": "Water-surface elevation",
            "profile_kind": "Profile",
            "state_label": "State",
        },
        title="Actual SWORD nodes versus matched state-node SWORD values versus monotone profile",
    )
    fig.update_traces(marker=dict(size=7))
    return fig


def plot_slope_plan_view(
    link_frame: pd.DataFrame,
    *,
    color_column: str = "slope_used",
    facet_by_example: bool = True,
) -> Any:
    if link_frame.empty:
        raise ValueError("No link-context data available for plotting.")
    if color_column not in link_frame.columns:
        raise ValueError(f"{color_column!r} is not present in link_frame.")
    required = {"centroid_x", "centroid_y"}
    if not required.issubset(link_frame.columns):
        raise ValueError("link_frame is missing centroid coordinates.")

    hover_cols = [
        column
        for column in [
            "example_id",
            "state_id",
            "id_link",
            "dominant_parent_link_id",
            "lineage_type",
            "slope_source_method",
            "raw_slope",
            "slope_used",
            "slope_section_ratio",
            "slope_neighbor_source_link_id",
            "wid_adj_routing",
            "wid_adj_wet",
        ]
        if column in link_frame.columns
    ]
    fig = px.scatter(
        link_frame,
        x="centroid_x",
        y="centroid_y",
        color=color_column,
        symbol="state_id",
        facet_col="example_id" if facet_by_example and link_frame["example_id"].nunique() > 1 else None,
        hover_data=hover_cols,
        labels={
            "centroid_x": "Projected X",
            "centroid_y": "Projected Y",
            color_column: color_column,
            "example_id": "Example ID",
            "state_id": "State ID",
        },
        title=f"Plan-view link diagnostics colored by {color_column}",
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_slope_ratio_scatter(
    link_frame: pd.DataFrame,
    *,
    x: str = "slope_section_ratio",
    y: str = "raw_slope",
    facet_by_example: bool = True,
) -> Any:
    if link_frame.empty:
        raise ValueError("No link-context data available for plotting.")
    for column in (x, y):
        if column not in link_frame.columns:
            raise ValueError(f"{column!r} is not present in link_frame.")

    hover_cols = [
        column
        for column in [
            "example_id",
            "state_id",
            "id_link",
            "dominant_parent_link_id",
            "lineage_type",
            "slope_source_method",
            "slope_used",
            "slope_neighbor_source_link_id",
        ]
        if column in link_frame.columns
    ]
    fig = px.scatter(
        link_frame,
        x=x,
        y=y,
        color="slope_source_method" if "slope_source_method" in link_frame.columns else None,
        symbol="state_id",
        facet_col="example_id" if facet_by_example and link_frame["example_id"].nunique() > 1 else None,
        hover_data=hover_cols,
        labels={
            "slope_section_ratio": "Local slope / section slope",
            "raw_slope": "Raw link slope",
            "slope_source_method": "Slope source",
            "example_id": "Example ID",
        },
        title="Raw slope versus section-ratio diagnostics",
    )
    return fig


__all__ = [
    "build_actual_sword_node_frame",
    "build_low_k_parent_summary",
    "build_slope_link_frame",
    "build_slope_node_frame",
    "build_slope_profile_long_frame",
    "identify_slope_seed_links",
    "list_slope_states",
    "plot_actual_sword_vs_state_vs_monotone",
    "plot_raw_vs_monotone_profile",
    "plot_slope_plan_view",
    "plot_slope_profiles",
    "plot_slope_ratio_scatter",
    "plot_sword_vs_monotone_profile",
]
