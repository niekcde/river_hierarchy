from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from hierarchy_level_definition.unit_detection.bifurcation_confluence_units import (
    build_graph,
    build_structural_units,
)
from network_variants.adaptive_routing_solver import (
    ADAPTIVE_ROUTING_SOLVER_VERSION,
    AdaptiveSolverConfig,
    solve_adaptive_two_ended,
)
from network_variants.width_junction_audit import WindowSpecification, calculate_endpoint_widths
from network_variants.width_qc_final_model import FINAL_FEATURE_COLUMNS, FINAL_MODEL_VERSION
from network_variants.width_review_archive import _sha256
from network_variants.width_review_gui import (
    QC_MEDIAN_FALLBACK_FIELD,
    QC_TRIMMED_FIELD,
    build_candidate_frame,
    compute_sample_qc_frame,
)
from network_variants.width_routing_solver import prepare_hybrid_solver_links


ROUTING_WIDTH_APPLICATION_VERSION = "routing-width-application-v1"
DEFAULT_FINAL_MODEL_DIR = Path(__file__).resolve().parent / "output" / "width_qc_final_model"
DEFAULT_WORKFLOW_MANIFEST_PATH = DEFAULT_FINAL_MODEL_DIR / "width_qc_final_workflow_manifest_v1.json"
DEFAULT_ENDPOINT_WINDOW = WindowSpecification(0.10, 250.0, 1000.0)
DEFAULT_CANDIDATE_LENGTH_OVER_WIDTH = 1.5
DRY_CORRIDOR_MEANINGFUL_FRACTION = 0.05
ROUTING_TOTAL_CORRIDOR_RELATIVE_TOLERANCE = 0.05
ROUTING_AUDIT_FILENAME = "routing_width_application.csv"
ROUTING_MANIFEST_FILENAME = "routing_width_application_manifest.json"


def _serialized_value_count(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    text = str(value).strip().strip("[]")
    if not text:
        return 0
    return sum(bool(part.strip()) for part in text.split(","))


def _resolved_artifact_path(path_value: str | Path, *, fallback_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()
    fallback = fallback_dir / path.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def _load_model_contract(
    workflow_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    workflow_path = Path(workflow_manifest_path).expanduser().resolve()
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    if workflow.get("model_version") != FINAL_MODEL_VERSION:
        raise ValueError(
            f"Expected model version {FINAL_MODEL_VERSION!r}, found {workflow.get('model_version')!r}."
        )
    selected_model = str(workflow["selected_model"])
    model_manifest_path = _resolved_artifact_path(
        workflow["files"]["model_manifest"], fallback_dir=workflow_path.parent / "model"
    )
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if list(model_manifest.get("feature_columns", [])) != list(FINAL_FEATURE_COLUMNS):
        raise ValueError("Selected model feature columns differ from the frozen production contract.")
    if model_manifest.get("provisional_candidate_model") != selected_model:
        raise ValueError("Workflow and model manifests select different model types.")
    bundle_path = _resolved_artifact_path(
        model_manifest["candidate_bundles"][selected_model], fallback_dir=model_manifest_path.parent
    )
    expected_hash = (model_manifest.get("candidate_bundle_sha256", {}) or {}).get(selected_model)
    if expected_hash and _sha256(bundle_path) != expected_hash:
        raise ValueError("Selected width-QC model bundle does not match its manifest hash.")
    return workflow, model_manifest, workflow_path, bundle_path


def _measured_family_frame(enriched_links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "wid_adj_wet" not in enriched_links.columns:
        raise ValueError("Width-family links do not contain a wet-width measurement.")
    measured = enriched_links.copy()
    measured["wid_adj_wet"] = pd.to_numeric(measured["wid_adj_wet"], errors="coerce")
    if measured["wid_adj_wet"].isna().any() or measured["wid_adj_wet"].le(0).any():
        bad = measured.loc[
            measured["wid_adj_wet"].isna() | measured["wid_adj_wet"].le(0), "id_link"
        ]
        raise ValueError(f"Routing-width input contains invalid wet widths for links: {bad.astype(int).tolist()}")
    return measured


def _ensure_current_unit_membership(
    links: gpd.GeoDataFrame,
    directed_nodes: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, str]:
    """Provide the current-network membership count used by candidate scope."""
    result = links.copy()
    if "n_units" in result.columns:
        values = pd.to_numeric(result["n_units"], errors="coerce")
        if values.notna().all():
            result["n_units"] = values.astype(int)
            return result, "provided_current_state_membership"

    if "id_nodes" not in result.columns:
        missing = {"id_us_node", "id_ds_node"}.difference(result.columns)
        if missing:
            raise ValueError(
                "Cannot derive current child unit membership; links are missing "
                f"columns: {sorted(missing)}"
            )
        result["id_nodes"] = [
            f"{int(upstream)},{int(downstream)}"
            for upstream, downstream in zip(result["id_us_node"], result["id_ds_node"])
        ]

    graph = build_graph(result, directed_nodes)
    units = build_structural_units(graph, max_path_cutoff=100, max_paths=5000)
    membership_count = {int(link_id): 0 for link_id in result["id_link"]}
    for unit in units:
        for _, _, link_id in unit.edge_set:
            normalized_link_id = int(link_id)
            membership_count[normalized_link_id] = membership_count.get(normalized_link_id, 0) + 1
    result["n_units"] = result["id_link"].astype(int).map(membership_count).fillna(0).astype(int)
    return result, "derived_from_current_child_topology"


def _model_features(
    raw_links: gpd.GeoDataFrame,
    samples: pd.DataFrame,
    solver_links: pd.DataFrame,
) -> pd.DataFrame:
    _, sample_qc = compute_sample_qc_frame(samples)
    candidates = build_candidate_frame(
        raw_links,
        sample_qc,
        width_field="wid_adj_wet",
        len_over_width_threshold=DEFAULT_CANDIDATE_LENGTH_OVER_WIDTH,
    )
    sample_count = raw_links.get(
        "wid_pix_total", pd.Series(pd.NA, index=raw_links.index, dtype="object")
    ).map(_serialized_value_count)
    invalid_count = pd.to_numeric(
        raw_links.get("n_invalid_width_samples", pd.Series(np.nan, index=raw_links.index)),
        errors="coerce",
    )
    invalid_fraction = invalid_count / sample_count.replace(0, np.nan)
    invalid_by_link = pd.Series(invalid_fraction.to_numpy(), index=raw_links["id_link"].astype(int)).to_dict()

    incoming_context = solver_links.groupby("id_ds_node")["routing_context_width_ds"].sum(min_count=1)
    outgoing_context = solver_links.groupby("id_us_node")["routing_context_width_us"].sum(min_count=1)
    hybrid_ratio = candidates["id_us_node"].map(outgoing_context) / candidates["id_us_node"].map(incoming_context)

    features = pd.DataFrame(index=candidates.index)
    features["id_link"] = candidates["id_link"].astype(int)
    features["id_us_node"] = candidates["id_us_node"].astype(int)
    features["id_ds_node"] = candidates["id_ds_node"].astype(int)
    features["width_model_candidate"] = candidates["candidate_flag"].fillna(False).astype(bool)
    features["length_over_raw_wet_width"] = pd.to_numeric(candidates["len_over_width"], errors="coerce")
    features["split_sensitive"] = candidates["split_sensitive"].astype("boolean")
    features["opposition_cos_trimmed_mean"] = pd.to_numeric(candidates[QC_TRIMMED_FIELD], errors="coerce")
    features["opposition_cos_trimmed_median"] = pd.to_numeric(
        candidates[QC_MEDIAN_FALLBACK_FIELD], errors="coerce"
    )
    features["invalid_sample_fraction"] = features["id_link"].map(invalid_by_link)
    features["upstream_continuity_ratio"] = pd.to_numeric(
        candidates["us_node_out_over_in"], errors="coerce"
    )
    features["upstream_continuity_abs_deviation"] = pd.to_numeric(candidates["us_dev"], errors="coerce")
    features["upstream_hybrid_continuity_abs_deviation"] = (hybrid_ratio - 1.0).abs()
    return features


def _add_dry_corridor_provenance(
    audit: pd.DataFrame,
    raw_links: pd.DataFrame,
) -> pd.DataFrame:
    """Add physical-corridor checks without changing measured or routing widths."""
    result = audit.copy()
    raw_by_id = raw_links.set_index(raw_links["id_link"].astype(int), drop=False)
    link_ids = result["id_link"].astype(int)
    for field in ("wid_adj_total", "wid_adj_dry", "dry_fraction_adj"):
        if field in raw_by_id.columns:
            result[field] = link_ids.map(raw_by_id[field])
        elif field not in result.columns:
            result[field] = np.nan

    wet = pd.to_numeric(result["wid_adj_wet"], errors="coerce")
    routing = pd.to_numeric(result["wid_adj_routing"], errors="coerce")
    total = pd.to_numeric(result["wid_adj_total"], errors="coerce")
    dry = pd.to_numeric(result["wid_adj_dry"], errors="coerce")
    stored_dry_fraction = pd.to_numeric(result["dry_fraction_adj"], errors="coerce")
    derived_dry_fraction = dry / total.where(total.gt(0))
    dry_fraction = stored_dry_fraction.where(stored_dry_fraction.notna(), derived_dry_fraction)

    result["dry_corridor_fraction"] = dry_fraction
    result["dry_corridor_meaningful"] = (
        dry_fraction.ge(DRY_CORRIDOR_MEANINGFUL_FRACTION)
        & dry.gt(0)
        & total.gt(0)
    ).fillna(False)
    result["routing_to_wet_width_ratio"] = routing / wet.where(wet.gt(0))
    result["routing_to_total_width_ratio"] = routing / total.where(total.gt(0))
    tolerance_factor = 1.0 + ROUTING_TOTAL_CORRIDOR_RELATIVE_TOLERANCE
    result["measured_wet_exceeds_total_corridor"] = (
        wet.gt(total * tolerance_factor) & total.gt(0)
    ).fillna(False)
    result["routing_exceeds_total_corridor"] = (
        routing.gt(total * tolerance_factor) & total.gt(0)
    ).fillna(False)
    result["routing_corridor_consistency_flag"] = (
        result["measured_wet_exceeds_total_corridor"]
        | result["routing_exceeds_total_corridor"]
    ).astype(bool)
    total_available = total.notna() & total.gt(0)
    result["routing_corridor_consistency_reason"] = np.select(
        [
            ~total_available,
            result["measured_wet_exceeds_total_corridor"],
            result["routing_exceeds_total_corridor"],
            result["dry_corridor_meaningful"],
        ],
        [
            "total_corridor_width_unavailable",
            "measured_wet_width_exceeds_total_corridor",
            "effective_routing_width_exceeds_total_corridor",
            "meaningful_dry_corridor_with_consistent_routing_width",
        ],
        default="no_meaningful_dry_corridor",
    )
    return result


def apply_routing_width_model(
    enriched_links: gpd.GeoDataFrame,
    link_width_samples: pd.DataFrame,
    directed_nodes: gpd.GeoDataFrame,
    *,
    example_id: str,
    variant_id: str,
    workflow_manifest_path: str | Path = DEFAULT_WORKFLOW_MANIFEST_PATH,
    apply_known_base_exclusions: bool = True,
    evaluation_link_ids: Collection[int] | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict[str, Any]]:
    """Classify and solve effective routing widths without changing measured families.

    When ``evaluation_link_ids`` is provided, only those links may receive a
    new model decision. Existing routing values on all other links are fixed
    context for the joint solver and their complete provenance is preserved.
    """
    raw_links = _measured_family_frame(enriched_links)
    raw_links, unit_membership_source = _ensure_current_unit_membership(raw_links, directed_nodes)
    all_link_ids = set(raw_links["id_link"].astype(int))
    if evaluation_link_ids is None:
        evaluated_ids = all_link_ids
        scoped_evaluation = False
    else:
        evaluated_ids = {int(link_id) for link_id in evaluation_link_ids}
        missing_evaluated_ids = sorted(evaluated_ids.difference(all_link_ids))
        if missing_evaluated_ids:
            raise ValueError(
                "Routing-width evaluation contains link IDs absent from the child network: "
                f"{missing_evaluated_ids}"
            )
        scoped_evaluation = True
    evaluation_mask_by_id = raw_links["id_link"].astype(int).isin(evaluated_ids)
    if scoped_evaluation:
        fixed = raw_links.loc[~evaluation_mask_by_id]
        if "wid_adj_routing" not in fixed.columns:
            raise ValueError("Scoped child evaluation requires inherited wid_adj_routing values.")
        invalid_fixed = pd.to_numeric(fixed["wid_adj_routing"], errors="coerce").isna()
        invalid_fixed |= pd.to_numeric(fixed["wid_adj_routing"], errors="coerce").le(0)
        if invalid_fixed.any():
            raise ValueError(
                "Scoped child evaluation found fixed links without valid inherited routing widths: "
                f"{fixed.loc[invalid_fixed, 'id_link'].astype(int).tolist()}"
            )
    workflow, model_manifest, workflow_path, bundle_path = _load_model_contract(workflow_manifest_path)
    bundle = joblib.load(bundle_path)
    if list(bundle.get("feature_columns", [])) != list(FINAL_FEATURE_COLUMNS):
        raise ValueError("Loaded width-QC bundle differs from the frozen feature contract.")

    endpoints = calculate_endpoint_widths(
        example_id=str(example_id).zfill(2),
        families=raw_links,
        samples=link_width_samples,
        directed_links=raw_links,
        directed_nodes=directed_nodes,
        windows=[DEFAULT_ENDPOINT_WINDOW],
        whole_width_field="wid_adj_wet",
    )
    solver_links = prepare_hybrid_solver_links(
        endpoints,
        length_over_width_threshold=10.0,
        require_nonoverlap=True,
    )
    if scoped_evaluation:
        raw_by_id = raw_links.set_index(raw_links["id_link"].astype(int), drop=False)
        solver_ids = solver_links["id_link"].astype(int)
        fixed_solver_mask = ~solver_ids.isin(evaluated_ids)
        fixed_wet = solver_ids.map(raw_by_id["wid_adj_wet"])
        fixed_routing = solver_ids.map(raw_by_id["wid_adj_routing"])
        fixed_scale = fixed_routing / fixed_wet
        # Endpoint contexts are physical measurements. Fixed inherited links
        # enter the child solver at their already accepted effective scale.
        for context_field in ("routing_context_width_us", "routing_context_width_ds"):
            solver_links.loc[fixed_solver_mask, context_field] = (
                pd.to_numeric(solver_links.loc[fixed_solver_mask, context_field], errors="coerce")
                * fixed_scale.loc[fixed_solver_mask]
            )
        feature_links = raw_links.copy()
        feature_links.loc[~evaluation_mask_by_id, "wid_adj_wet"] = pd.to_numeric(
            feature_links.loc[~evaluation_mask_by_id, "wid_adj_routing"], errors="coerce"
        )
    else:
        feature_links = raw_links
    features = _model_features(feature_links, link_width_samples, solver_links)
    features["width_model_evaluation_scope"] = features["id_link"].isin(evaluated_ids)
    features["width_model_excluded"] = False
    if apply_known_base_exclusions:
        excluded_path = _resolved_artifact_path(
            workflow["files"]["excluded_cases"], fallback_dir=workflow_path.parent
        )
        excluded = pd.read_csv(excluded_path, dtype={"example_id": "string"})
        excluded["example_id"] = excluded["example_id"].astype("string").str.zfill(2)
        excluded_ids = set(
            pd.to_numeric(
                excluded.loc[excluded["example_id"].eq(str(example_id).zfill(2)), "id_link"],
                errors="coerce",
            ).dropna().astype(int)
        )
        features["width_model_excluded"] = features["id_link"].isin(excluded_ids)

    threshold = float(workflow["selected_threshold"])
    predicted_scope = (
        features["width_model_evaluation_scope"]
        & features["width_model_candidate"]
        & ~features["width_model_excluded"]
    )
    features["width_model_probability"] = np.nan
    if predicted_scope.any():
        probabilities = bundle["pipeline"].predict_proba(
            features.loc[predicted_scope, list(FINAL_FEATURE_COLUMNS)]
        )[:, 1]
        features.loc[predicted_scope, "width_model_probability"] = probabilities
    features["width_model_threshold"] = threshold
    features["width_model_prediction"] = (
        predicted_scope & features["width_model_probability"].ge(threshold)
    ).astype(int)

    solver_input = solver_links.merge(
        features[
            [
                "id_link",
                "width_model_prediction",
                "width_model_probability",
                "width_model_threshold",
                "upstream_continuity_abs_deviation",
                "opposition_cos_trimmed_mean",
            ]
        ],
        on="id_link",
        how="left",
        validate="1:1",
    )
    solved, solver_diagnostics = solve_adaptive_two_ended(
        solver_input,
        config=AdaptiveSolverConfig(upstream_weight=2.0, downstream_weight=1.0),
    )
    solved_fields = solved[
        [
            "id_link",
            "routing_context_width_us",
            "routing_context_width_ds",
            "routing_context_uses_endpoint",
            "comparison_scale_factor",
            "comparison_routing_width",
            "comparison_status",
            "comparison_flag",
            "model_confidence_strength",
            "upstream_deviation_strength",
            "opposition_badness",
            "adaptive_relaxation_strength",
            "adaptive_min_scale",
            "adaptive_max_scale",
            "adaptive_raw_penalty_weight",
        ]
    ]
    audit = features.merge(solved_fields, on="id_link", how="left", validate="1:1")
    audit["example_id"] = str(example_id).zfill(2)
    audit["variant_id"] = str(variant_id)
    audit["wid_adj_wet"] = audit["id_link"].map(
        raw_links.set_index("id_link")["wid_adj_wet"].to_dict()
    )
    audit["wid_adj_routing"] = audit["comparison_routing_width"]
    audit["width_correction_scale_factor"] = audit["comparison_scale_factor"]
    audit["width_correction_applied"] = audit["width_model_prediction"].eq(1) & ~np.isclose(
        audit["width_correction_scale_factor"], 1.0
    )
    audit = _add_dry_corridor_provenance(audit, raw_links)
    audit["width_correction_flag"] = (
        audit["comparison_flag"].fillna(False).astype(bool)
        | audit["routing_corridor_consistency_flag"].fillna(False).astype(bool)
    )
    audit["width_qc_decision"] = np.select(
        [
            audit["width_model_excluded"],
            audit["width_correction_flag"],
            audit["width_model_prediction"].eq(1),
        ],
        ["retain_raw_flag", "correct_flag", "correct"],
        default="accept",
    )
    audit["width_qc_reason"] = np.select(
        [
            audit["width_model_excluded"],
            audit["routing_corridor_consistency_flag"],
            ~audit["width_model_candidate"],
            audit["width_correction_flag"],
            audit["width_model_prediction"].eq(1),
        ],
        [
            "known_ambiguous_review_case",
            "effective_width_inconsistent_with_measured_corridor",
            "outside_model_candidate_scope",
            "model_correction_outside_normal_scale_bounds",
            "model_predicted_correction",
        ],
        default="model_accepted_measurement",
    )
    audit["width_correction_absolute_change"] = (
        audit["wid_adj_routing"] - audit["wid_adj_wet"]
    )
    audit["width_correction_relative_change"] = (
        audit["width_correction_absolute_change"] / audit["wid_adj_wet"]
    )
    audit["width_correction_source"] = ADAPTIVE_ROUTING_SOLVER_VERSION
    audit["width_algorithm_version"] = ROUTING_WIDTH_APPLICATION_VERSION
    audit["width_model_version"] = str(bundle["model_version"])
    audit["width_model_name"] = str(bundle["model_name"])

    if scoped_evaluation:
        fixed_audit_mask = ~audit["id_link"].astype(int).isin(evaluated_ids)
        raw_by_id = raw_links.set_index(raw_links["id_link"].astype(int), drop=False)
        # Replace provisional calculations on fixed links with their inherited
        # parent record. The audit is therefore a complete final child table,
        # not a mixture of inherited and accidentally recalculated values.
        for column in audit.columns:
            if column in {"id_link", "example_id", "variant_id", "width_model_evaluation_scope"}:
                continue
            if column not in raw_by_id.columns:
                continue
            inherited_values = audit.loc[fixed_audit_mask, "id_link"].astype(int).map(raw_by_id[column])
            if pd.api.types.is_bool_dtype(audit[column].dtype):
                assigned_values = inherited_values.astype("boolean").to_numpy(
                    dtype=bool,
                    na_value=False,
                )
            else:
                assigned_values = inherited_values.to_numpy()
            audit.loc[fixed_audit_mask, column] = assigned_values
        audit["width_evaluation_status"] = np.where(
            audit["width_model_evaluation_scope"],
            "evaluated_changed_link",
            "preserved_inherited_link",
        )
    else:
        audit["width_evaluation_status"] = "evaluated_base_link"

    output = enriched_links.copy()
    output["n_units"] = output["id_link"].astype(int).map(
        raw_links.set_index(raw_links["id_link"].astype(int))["n_units"]
    ).fillna(0).astype(int)
    audit_by_id = audit.set_index(audit["id_link"].astype(int), drop=False)
    output_ids = output["id_link"].astype(int)
    output_evaluation_mask = output_ids.isin(evaluated_ids)
    current_state_fields = {"width_model_evaluation_scope", "width_evaluation_status"}
    for column in audit.columns:
        if column in {"id_link", "example_id", "variant_id"}:
            continue
        mapped = output_ids.map(audit_by_id[column])
        if column not in output.columns or column in current_state_fields:
            output[column] = mapped
        elif output_evaluation_mask.any():
            output.loc[output_evaluation_mask, column] = mapped.loc[output_evaluation_mask].to_numpy()
    # Preserve the state identifiers already attached by the materializer.
    output["example_id"] = str(example_id)
    output["variant_id"] = str(variant_id)

    summary = {
        "schema_version": ROUTING_WIDTH_APPLICATION_VERSION,
        "example_id": str(example_id).zfill(2),
        "variant_id": str(variant_id),
        "measurement_source_field": "wid_adj_wet",
        "unit_membership_source": unit_membership_source,
        "model_version": str(bundle["model_version"]),
        "model_name": str(bundle["model_name"]),
        "model_threshold": threshold,
        "model_bundle": str(bundle_path),
        "model_bundle_sha256": _sha256(bundle_path),
        "solver_version": ADAPTIVE_ROUTING_SOLVER_VERSION,
        "solver_weights": {"upstream": 2.0, "downstream": 1.0},
        "dry_corridor_meaningful_fraction": DRY_CORRIDOR_MEANINGFUL_FRACTION,
        "routing_total_corridor_relative_tolerance": ROUTING_TOTAL_CORRIDOR_RELATIVE_TOLERANCE,
        "link_count": int(len(output)),
        "evaluation_scope": "selected_changed_links" if scoped_evaluation else "all_links",
        "evaluated_link_count": int(audit["width_model_evaluation_scope"].sum()),
        "fixed_inherited_link_count": int((~audit["width_model_evaluation_scope"]).sum()),
        "candidate_count": int(
            (audit["width_model_evaluation_scope"] & audit["width_model_candidate"]).sum()
        ),
        "excluded_count": int(
            (audit["width_model_evaluation_scope"] & audit["width_model_excluded"]).sum()
        ),
        "predicted_correction_count": int(
            audit.loc[audit["width_model_evaluation_scope"], "width_model_prediction"].sum()
        ),
        "applied_correction_count": int(
            audit.loc[audit["width_model_evaluation_scope"], "width_correction_applied"].sum()
        ),
        "flagged_correction_count": int(
            audit.loc[audit["width_model_evaluation_scope"], "width_correction_flag"].sum()
        ),
        "meaningful_dry_corridor_count": int(
            audit.loc[audit["width_model_evaluation_scope"], "dry_corridor_meaningful"].sum()
        ),
        "corridor_consistency_flag_count": int(
            audit.loc[
                audit["width_model_evaluation_scope"],
                "routing_corridor_consistency_flag",
            ].sum()
        ),
        "endpoint_context_link_count": int(
            audit.loc[audit["width_model_evaluation_scope"], "routing_context_uses_endpoint"].sum()
        ),
        "solver_diagnostics": solver_diagnostics,
    }
    return output, audit, summary


def write_routing_width_audit(
    audit: pd.DataFrame,
    summary: dict[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / ROUTING_AUDIT_FILENAME
    manifest_path = output / ROUTING_MANIFEST_FILENAME
    audit.to_csv(audit_path, index=False)
    manifest = dict(summary)
    # Keep nested artifact references relative so copied anchored/base states
    # remain self-contained and do not depend on the realized source tree.
    manifest["files"] = {"link_audit": audit_path.name}
    manifest["file_sha256"] = {"link_audit": _sha256(audit_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"link_audit": str(audit_path.resolve()), "manifest": str(manifest_path.resolve())}
