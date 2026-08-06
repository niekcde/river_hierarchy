from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from network_variants.width_junction_audit import (
    DEFAULT_OUTPUT_DIR as DEFAULT_JUNCTION_AUDIT_DIR,
    ENDPOINTS_FILENAME,
    MANIFEST_FILENAME as JUNCTION_MANIFEST_FILENAME,
)
from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256
from network_variants.width_review_dataset import (
    DATASET_FILENAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_DATASET_DIR,
)


HYBRID_AUDIT_SCHEMA_VERSION = "width-junction-hybrid-audit-v1"
DEFAULT_WINDOW_ID = "p10_min250_max1000"
DEFAULT_REVIEW_DIR = Path(__file__).resolve().parent / "output" / "width_qc_disagreement_review"
DEFAULT_REVIEW_PATH = DEFAULT_REVIEW_DIR / "width_qc_disagreement_review_v1.csv"
DEFAULT_REVIEW_MANIFEST_PATH = DEFAULT_REVIEW_DIR / "width_qc_disagreement_review_manifest_v1.json"
FROZEN_REVIEW_REBASE_FILENAME = "width_qc_disagreement_review_rebase_v1.json"
DEFAULT_DATASET_PATH = DEFAULT_DATASET_DIR / DATASET_FILENAME
DEFAULT_ENDPOINTS_PATH = DEFAULT_JUNCTION_AUDIT_DIR / ENDPOINTS_FILENAME
DEFAULT_JUNCTION_MANIFEST_PATH = DEFAULT_JUNCTION_AUDIT_DIR / JUNCTION_MANIFEST_FILENAME
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_qc_junction_hybrid_audit"

COMPARISONS_FILENAME = "hybrid_junction_comparisons_v1.csv"
PROFILE_FILENAME = "hybrid_junction_variant_profile_v1.csv"
AMBIGUOUS_FILENAME = "hybrid_ambiguous_cases_v1.csv"
FOLLOWUP_FILENAME = "hybrid_followup_candidates_v1.csv"
MANIFEST_FILENAME = "hybrid_junction_audit_manifest_v1.json"


@dataclass(frozen=True)
class HybridVariant:
    variant_id: str
    length_over_width_threshold: float | None
    require_nonoverlap: bool
    endpoint_mode: str

    def eligible(self, links: pd.DataFrame) -> pd.Series:
        available = (
            pd.to_numeric(links["width_near_us_node_wet"], errors="coerce").notna()
            & pd.to_numeric(links["width_near_ds_node_wet"], errors="coerce").notna()
        )
        if self.endpoint_mode == "whole_mean":
            return pd.Series(False, index=links.index)
        if self.endpoint_mode == "endpoint_all":
            return available
        if self.endpoint_mode != "threshold" or self.length_over_width_threshold is None:
            raise ValueError(f"Unsupported hybrid endpoint mode: {self.endpoint_mode}")
        eligible = pd.to_numeric(links["length_over_wet_width"], errors="coerce").ge(
            self.length_over_width_threshold
        )
        if self.require_nonoverlap:
            eligible &= links["endpoint_windows_nonoverlapping"].astype("boolean").fillna(False)
        return available & eligible


DEFAULT_VARIANTS = (
    HybridVariant("whole_mean", None, False, "whole_mean"),
    HybridVariant("endpoint_all", None, False, "endpoint_all"),
    HybridVariant("lw_ge_3", 3.0, False, "threshold"),
    HybridVariant("lw_ge_5", 5.0, False, "threshold"),
    HybridVariant("lw_ge_10", 10.0, False, "threshold"),
    HybridVariant("lw_ge_3_nonoverlap", 3.0, True, "threshold"),
    HybridVariant("lw_ge_5_nonoverlap", 5.0, True, "threshold"),
    HybridVariant("lw_ge_10_nonoverlap", 10.0, True, "threshold"),
)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _serialize_ids(values: pd.Series) -> str:
    return ",".join(str(int(value)) for value in sorted(pd.to_numeric(values, errors="coerce").dropna()))


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float | None:
    mask = labels.notna() & scores.notna()
    y = labels.loc[mask].astype(int)
    if y.nunique() != 2:
        return None
    return float(roc_auc_score(y, scores.loc[mask]))


def _safe_average_precision(labels: pd.Series, scores: pd.Series) -> float | None:
    mask = labels.notna() & scores.notna()
    y = labels.loc[mask].astype(int)
    if not y.eq(1).any():
        return None
    return float(average_precision_score(y, scores.loc[mask]))


def build_effective_label_frame(dataset: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    required_dataset = {
        "example_id",
        "id_link",
        "manual_decision",
        "requires_correction",
        "split_sensitive",
    }
    missing = sorted(required_dataset.difference(dataset.columns))
    if missing:
        raise ValueError(f"Training dataset is missing hybrid-audit fields: {missing}")
    required_review = {
        "example_id",
        "id_link",
        "adjudication",
        "adjudicated_requires_correction",
        "include_in_training",
        "review_comment",
    }
    missing = sorted(required_review.difference(review.columns))
    if missing:
        raise ValueError(f"Completed adjudication review is missing fields: {missing}")

    base = dataset[
        ["example_id", "id_link", "manual_decision", "requires_correction", "split_sensitive"]
    ].copy()
    base["example_id"] = base["example_id"].astype("string").str.zfill(2)
    base["id_link"] = pd.to_numeric(base["id_link"], errors="raise").astype(int)
    decisions = review[list(required_review)].copy()
    decisions["example_id"] = decisions["example_id"].astype("string").str.zfill(2)
    decisions["id_link"] = pd.to_numeric(decisions["id_link"], errors="raise").astype(int)
    if decisions.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Completed adjudication review contains duplicate example/link keys.")
    frame = base.merge(decisions, on=["example_id", "id_link"], how="left", validate="1:1")
    frame["effective_requires_correction"] = frame["requires_correction"].astype("Int64")
    frame["effective_label_source"] = "original_manual_nonqueue"

    manual_upheld = frame["adjudication"].eq("manual_upheld")
    model_upheld = frame["adjudication"].eq("model_upheld")
    resolved = manual_upheld | model_upheld | frame["adjudication"].eq(
        "measurement_valid_but_routing_unsuitable"
    )
    frame.loc[resolved, "effective_requires_correction"] = pd.to_numeric(
        frame.loc[resolved, "adjudicated_requires_correction"], errors="coerce"
    ).astype("Int64")
    frame.loc[manual_upheld, "effective_label_source"] = "adjudicated_manual_upheld"
    frame.loc[model_upheld, "effective_label_source"] = "adjudicated_model_upheld"
    frame.loc[
        frame["adjudication"].eq("measurement_valid_but_routing_unsuitable"),
        "effective_label_source",
    ] = "adjudicated_routing_unsuitable"

    unresolved = frame["adjudication"].isin(
        ["ambiguous", "data_or_geometry_problem", "exclude_from_training"]
    )
    frame.loc[unresolved, "effective_requires_correction"] = pd.NA
    frame.loc[unresolved, "effective_label_source"] = (
        "excluded_" + frame.loc[unresolved, "adjudication"].astype(str)
    )
    frame["in_completed_review"] = frame["adjudication"].notna()
    return frame


def calculate_hybrid_comparisons(
    endpoint_widths: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    variants: Sequence[HybridVariant] = DEFAULT_VARIANTS,
) -> pd.DataFrame:
    required = {
        "example_id",
        "id_link",
        "id_us_node",
        "id_ds_node",
        "whole_link_mean_wet_width",
        "width_near_us_node_wet",
        "width_near_ds_node_wet",
        "length_over_wet_width",
        "endpoint_windows_nonoverlapping",
        "window_id",
    }
    missing = sorted(required.difference(endpoint_widths.columns))
    if missing:
        raise ValueError(f"Endpoint-width audit is missing hybrid fields: {missing}")
    if not variants or len({variant.variant_id for variant in variants}) != len(variants):
        raise ValueError("Hybrid variants must be non-empty and uniquely named.")

    endpoints = endpoint_widths.copy()
    endpoints["example_id"] = endpoints["example_id"].astype("string").str.zfill(2)
    for column in ("id_link", "id_us_node", "id_ds_node"):
        endpoints[column] = pd.to_numeric(endpoints[column], errors="raise").astype(int)
    labels = label_frame.copy()
    labels["example_id"] = labels["example_id"].astype("string").str.zfill(2)
    labels["id_link"] = pd.to_numeric(labels["id_link"], errors="raise").astype(int)

    endpoint_keys = endpoints[["example_id", "id_link"]].drop_duplicates()
    missing_label_links = labels.merge(
        endpoint_keys,
        on=["example_id", "id_link"],
        how="left",
        indicator=True,
    ).loc[lambda frame: frame["_merge"].eq("left_only"), ["example_id", "id_link"]]
    if not missing_label_links.empty:
        raise ValueError(
            "Endpoint-width audit is missing reviewed links: "
            f"{missing_label_links.to_dict('records')[:10]}"
        )

    output: list[pd.DataFrame] = []
    for example_id, links in endpoints.groupby("example_id", sort=True):
        links = links.copy()
        if links.duplicated("id_link").any():
            raise ValueError(f"Endpoint widths contain duplicate links for example {example_id}.")
        reviewed = labels.loc[labels["example_id"].eq(example_id)].copy()
        upstream_node = reviewed["id_link"].map(links.set_index("id_link")["id_us_node"])
        if upstream_node.isna().any():
            missing_ids = reviewed.loc[upstream_node.isna(), "id_link"].tolist()
            raise ValueError(f"Example {example_id} is missing endpoint links: {missing_ids}")

        for variant in variants:
            eligible = variant.eligible(links)
            working = links.copy()
            working["uses_endpoint"] = eligible
            working["selected_us_width"] = np.where(
                eligible,
                working["width_near_us_node_wet"],
                working["whole_link_mean_wet_width"],
            )
            working["selected_ds_width"] = np.where(
                eligible,
                working["width_near_ds_node_wet"],
                working["whole_link_mean_wet_width"],
            )
            incoming_sum = working.groupby("id_ds_node")["selected_ds_width"].sum(min_count=1)
            outgoing_sum = working.groupby("id_us_node")["selected_us_width"].sum(min_count=1)
            incoming_ids = working.groupby("id_ds_node")["id_link"].apply(_serialize_ids)
            outgoing_ids = working.groupby("id_us_node")["id_link"].apply(_serialize_ids)
            incoming_endpoint_ids = working.loc[working["uses_endpoint"]].groupby("id_ds_node")[
                "id_link"
            ].apply(_serialize_ids)
            outgoing_endpoint_ids = working.loc[working["uses_endpoint"]].groupby("id_us_node")[
                "id_link"
            ].apply(_serialize_ids)
            incoming_endpoint_count = working.loc[working["uses_endpoint"]].groupby("id_ds_node")[
                "id_link"
            ].count()
            outgoing_endpoint_count = working.loc[working["uses_endpoint"]].groupby("id_us_node")[
                "id_link"
            ].count()

            result = reviewed.copy()
            result["hybrid_variant_id"] = variant.variant_id
            result["incoming_link_ids"] = upstream_node.map(incoming_ids)
            result["outgoing_link_ids"] = upstream_node.map(outgoing_ids)
            result["incoming_endpoint_link_ids"] = upstream_node.map(incoming_endpoint_ids).fillna("")
            result["outgoing_endpoint_link_ids"] = upstream_node.map(outgoing_endpoint_ids).fillna("")
            result["incoming_endpoint_link_count"] = upstream_node.map(incoming_endpoint_count).fillna(0).astype(int)
            result["outgoing_endpoint_link_count"] = upstream_node.map(outgoing_endpoint_count).fillna(0).astype(int)
            result["hybrid_incoming_width_sum"] = upstream_node.map(incoming_sum)
            result["hybrid_outgoing_width_sum"] = upstream_node.map(outgoing_sum)
            result["hybrid_continuity_ratio"] = (
                result["hybrid_outgoing_width_sum"] / result["hybrid_incoming_width_sum"]
            )
            result["hybrid_continuity_abs_deviation"] = (
                result["hybrid_continuity_ratio"] - 1.0
            ).abs()
            output.append(result)
    comparisons = pd.concat(output, ignore_index=True)
    expected_rows = len(labels) * len(variants)
    if len(comparisons) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} hybrid comparison rows, found {len(comparisons)}."
        )
    variant_by_id = {variant.variant_id: variant for variant in variants}
    comparisons["length_over_width_threshold"] = comparisons["hybrid_variant_id"].map(
        lambda variant_id: variant_by_id[variant_id].length_over_width_threshold
    )
    comparisons["require_nonoverlap"] = comparisons["hybrid_variant_id"].map(
        lambda variant_id: variant_by_id[variant_id].require_nonoverlap
    )
    comparisons["endpoint_mode"] = comparisons["hybrid_variant_id"].map(
        lambda variant_id: variant_by_id[variant_id].endpoint_mode
    )
    baseline = comparisons.loc[
        comparisons["hybrid_variant_id"].eq("whole_mean"),
        ["example_id", "id_link", "hybrid_continuity_abs_deviation"],
    ].rename(columns={"hybrid_continuity_abs_deviation": "whole_mean_abs_deviation"})
    endpoint_all = comparisons.loc[
        comparisons["hybrid_variant_id"].eq("endpoint_all"),
        ["example_id", "id_link", "hybrid_continuity_abs_deviation"],
    ].rename(columns={"hybrid_continuity_abs_deviation": "endpoint_all_abs_deviation"})
    comparisons = comparisons.merge(baseline, on=["example_id", "id_link"], validate="m:1")
    comparisons = comparisons.merge(endpoint_all, on=["example_id", "id_link"], validate="m:1")
    comparisons["change_from_whole_mean"] = (
        comparisons["hybrid_continuity_abs_deviation"] - comparisons["whole_mean_abs_deviation"]
    )
    comparisons["change_from_endpoint_all"] = (
        comparisons["hybrid_continuity_abs_deviation"] - comparisons["endpoint_all_abs_deviation"]
    )
    return comparisons.sort_values(
        ["hybrid_variant_id", "example_id", "id_link"], kind="mergesort"
    ).reset_index(drop=True)


def build_variant_profile(
    comparisons: pd.DataFrame,
    endpoint_widths: pd.DataFrame,
    *,
    variants: Sequence[HybridVariant],
) -> pd.DataFrame:
    profile_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows = comparisons.loc[comparisons["hybrid_variant_id"].eq(variant.variant_id)].copy()
        endpoint_eligible = variant.eligible(endpoint_widths)
        effective = rows["effective_requires_correction"]
        scores = rows["hybrid_continuity_abs_deviation"]
        resolved_rows = rows.loc[
            rows["in_completed_review"] & rows["effective_requires_correction"].notna()
        ]
        profile_rows.append(
            {
                "hybrid_variant_id": variant.variant_id,
                "endpoint_mode": variant.endpoint_mode,
                "length_over_width_threshold": variant.length_over_width_threshold,
                "require_nonoverlap": variant.require_nonoverlap,
                "unique_links_using_endpoint_width": int(endpoint_eligible.sum()),
                "unique_link_fraction_using_endpoint_width": float(endpoint_eligible.mean()),
                "reviewed_link_count": int(len(rows)),
                "effective_label_count": int(effective.notna().sum()),
                "excluded_ambiguous_or_problem_count": int(effective.isna().sum()),
                "effective_label_roc_auc": _safe_auc(effective, scores),
                "effective_label_average_precision": _safe_average_precision(effective, scores),
                "resolved_adjudication_count": int(len(resolved_rows)),
                "resolved_adjudication_roc_auc": _safe_auc(
                    resolved_rows["effective_requires_correction"],
                    resolved_rows["hybrid_continuity_abs_deviation"],
                ),
                "resolved_adjudication_average_precision": _safe_average_precision(
                    resolved_rows["effective_requires_correction"],
                    resolved_rows["hybrid_continuity_abs_deviation"],
                ),
                "median_abs_deviation_effective_good": float(
                    rows.loc[effective.eq(0), "hybrid_continuity_abs_deviation"].median()
                ),
                "median_abs_deviation_effective_bad": float(
                    rows.loc[effective.eq(1), "hybrid_continuity_abs_deviation"].median()
                ),
                "material_changes_from_whole_mean": int(rows["change_from_whole_mean"].abs().ge(0.25).sum()),
                "material_changes_from_endpoint_all": int(rows["change_from_endpoint_all"].abs().ge(0.25).sum()),
                "median_absolute_change_from_whole_mean": float(rows["change_from_whole_mean"].abs().median()),
                "p95_absolute_change_from_whole_mean": float(rows["change_from_whole_mean"].abs().quantile(0.95)),
            }
        )
    return pd.DataFrame(profile_rows)


def build_followup_candidates(comparisons: pd.DataFrame) -> pd.DataFrame:
    keys = ["example_id", "id_link"]
    metadata_columns = [
        "example_id",
        "id_link",
        "manual_decision",
        "requires_correction",
        "adjudication",
        "adjudicated_requires_correction",
        "effective_requires_correction",
        "effective_label_source",
        "review_comment",
    ]
    metadata = comparisons[metadata_columns].drop_duplicates(keys)
    if metadata.duplicated(keys).any():
        raise ValueError("Hybrid comparisons contain inconsistent per-link metadata.")
    wide_values = comparisons.pivot(
        index=keys,
        columns="hybrid_variant_id",
        values="hybrid_continuity_abs_deviation",
    ).reset_index()
    wide_values.columns.name = None
    wide = metadata.merge(wide_values, on=keys, how="inner", validate="1:1")
    variant_columns = [variant.variant_id for variant in DEFAULT_VARIANTS]
    missing_variants = sorted(set(variant_columns).difference(wide.columns))
    if missing_variants:
        raise ValueError(f"Hybrid comparisons are missing variants: {missing_variants}")
    hybrid_columns = [column for column in variant_columns if column not in {"whole_mean", "endpoint_all"}]
    wide["hybrid_rule_deviation_min"] = wide[hybrid_columns].min(axis=1)
    wide["hybrid_rule_deviation_max"] = wide[hybrid_columns].max(axis=1)
    wide["hybrid_rule_deviation_range"] = (
        wide["hybrid_rule_deviation_max"] - wide["hybrid_rule_deviation_min"]
    )
    ambiguous = wide["adjudication"].eq("ambiguous")
    resolved_sensitive = (
        wide["adjudication"].isin(["manual_upheld", "model_upheld"])
        & wide["hybrid_rule_deviation_range"].ge(0.25)
    )
    wide["followup_reason"] = np.select(
        [ambiguous, resolved_sensitive],
        ["ambiguous_adjudication", "resolved_adjudication_sensitive_to_hybrid_rule"],
        default="",
    )
    wide["followup_priority"] = np.select([ambiguous, resolved_sensitive], [1, 2], default=3)
    selected = wide.loc[ambiguous | resolved_sensitive].copy()
    return selected.sort_values(
        ["followup_priority", "hybrid_rule_deviation_range", "example_id", "id_link"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_source_manifests(
    *,
    dataset_path: Path,
    endpoints_path: Path,
    junction_manifest_path: Path,
    review_path: Path,
    review_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    junction_manifest = json.loads(junction_manifest_path.read_text(encoding="utf-8"))
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    dataset_hash = _sha256(dataset_path)
    endpoints_hash = _sha256(endpoints_path)
    review_hash = _sha256(review_path)
    if junction_manifest.get("dataset_sha256") != dataset_hash:
        raise ValueError("Junction audit and training dataset hashes disagree.")
    expected_endpoints = (junction_manifest.get("file_sha256", {}) or {}).get("endpoint_widths")
    if expected_endpoints != endpoints_hash:
        raise ValueError("Endpoint-width CSV does not match its junction-audit manifest.")
    expected_review = (review_manifest.get("file_sha256", {}) or {}).get("review")
    if expected_review != review_hash:
        raise ValueError("Adjudication CSV does not match its review manifest.")
    original_review_dataset_hash = (review_manifest.get("inputs", {}) or {}).get("dataset_sha256")
    if original_review_dataset_hash != dataset_hash:
        rebase_path = review_manifest_path.parent / FROZEN_REVIEW_REBASE_FILENAME
        if not rebase_path.exists():
            raise ValueError("Adjudication review and training dataset hashes disagree.")
        rebase = json.loads(rebase_path.read_text(encoding="utf-8"))
        if rebase.get("status") != "validated_frozen_adjudication_reuse":
            raise ValueError("Frozen adjudication rebase is not validated.")
        if (rebase.get("original_inputs", {}) or {}).get("dataset_sha256") != original_review_dataset_hash:
            raise ValueError("Frozen adjudication rebase does not originate from the reviewed dataset.")
        if (rebase.get("current_inputs", {}) or {}).get("dataset_sha256") != dataset_hash:
            raise ValueError("Frozen adjudication rebase does not authorize the current dataset.")
        human_review = rebase.get("human_review", {}) or {}
        if human_review.get("review_sha256") != review_hash:
            raise ValueError("Frozen adjudication rebase refers to a different human-review CSV.")
        if human_review.get("manifest_sha256") != _sha256(review_manifest_path):
            raise ValueError("Frozen adjudication rebase refers to a different human-review manifest.")
    if int((review_manifest.get("counts", {}) or {}).get("unreviewed", -1)) != 0:
        raise ValueError("Adjudication review is incomplete.")
    return junction_manifest, review_manifest


def run_hybrid_junction_audit(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    endpoints_path: str | Path = DEFAULT_ENDPOINTS_PATH,
    junction_manifest_path: str | Path = DEFAULT_JUNCTION_MANIFEST_PATH,
    review_path: str | Path = DEFAULT_REVIEW_PATH,
    review_manifest_path: str | Path = DEFAULT_REVIEW_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    window_id: str = DEFAULT_WINDOW_ID,
    variants: Sequence[HybridVariant] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    endpoints_path = Path(endpoints_path).expanduser().resolve()
    junction_manifest_path = Path(junction_manifest_path).expanduser().resolve()
    review_path = Path(review_path).expanduser().resolve()
    review_manifest_path = Path(review_manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (
        dataset_path,
        endpoints_path,
        junction_manifest_path,
        review_path,
        review_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    junction_manifest, review_manifest = _validate_source_manifests(
        dataset_path=dataset_path,
        endpoints_path=endpoints_path,
        junction_manifest_path=junction_manifest_path,
        review_path=review_path,
        review_manifest_path=review_manifest_path,
    )
    dataset = pd.read_csv(dataset_path, dtype={"example_id": "string"})
    review = pd.read_csv(review_path, dtype={"example_id": "string"})
    endpoints = pd.read_csv(endpoints_path, dtype={"example_id": "string"})
    endpoints["example_id"] = endpoints["example_id"].str.zfill(2)
    endpoints = endpoints.loc[endpoints["window_id"].eq(window_id)].copy()
    if endpoints.empty:
        raise ValueError(f"Endpoint audit contains no rows for window {window_id!r}.")

    labels = build_effective_label_frame(dataset, review)
    comparisons = calculate_hybrid_comparisons(endpoints, labels, variants=variants)
    profile = build_variant_profile(comparisons, endpoints, variants=variants)
    ambiguous = comparisons.loc[comparisons["adjudication"].eq("ambiguous")].copy()
    followup = build_followup_candidates(comparisons)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparisons": output_dir / COMPARISONS_FILENAME,
        "profile": output_dir / PROFILE_FILENAME,
        "ambiguous": output_dir / AMBIGUOUS_FILENAME,
        "followup": output_dir / FOLLOWUP_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    _atomic_write_csv(comparisons, paths["comparisons"])
    _atomic_write_csv(profile, paths["profile"])
    _atomic_write_csv(ambiguous, paths["ambiguous"])
    _atomic_write_csv(followup, paths["followup"])
    manifest = {
        "schema_version": HYBRID_AUDIT_SCHEMA_VERSION,
        "code_revision": _git_revision(),
        "production_authorized": False,
        "window_id": window_id,
        "variant_contract": [
            {
                "hybrid_variant_id": variant.variant_id,
                "endpoint_mode": variant.endpoint_mode,
                "length_over_width_threshold": variant.length_over_width_threshold,
                "require_nonoverlap": variant.require_nonoverlap,
            }
            for variant in variants
        ],
        "label_contract": {
            "nonqueue": "retain original permanent manual label",
            "manual_upheld": "retain original manual label",
            "model_upheld": "use human-adjudicated model label",
            "ambiguous_or_problem": "exclude from evaluation pending follow-up",
            "caveat": "Model predictions were visible during adjudication; metrics are exploratory, not independent validation.",
        },
        "inputs": {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "endpoint_widths": str(endpoints_path),
            "endpoint_widths_sha256": _sha256(endpoints_path),
            "junction_manifest": str(junction_manifest_path),
            "junction_manifest_sha256": _sha256(junction_manifest_path),
            "adjudication_review": str(review_path),
            "adjudication_review_sha256": _sha256(review_path),
            "adjudication_manifest": str(review_manifest_path),
            "adjudication_manifest_sha256": _sha256(review_manifest_path),
            "junction_audit_code_revision": junction_manifest.get("code_revision"),
            "review_model_name": review_manifest.get("model_name"),
            "frozen_review_rebase": str(
                (review_manifest_path.parent / FROZEN_REVIEW_REBASE_FILENAME).resolve()
            )
            if (review_manifest_path.parent / FROZEN_REVIEW_REBASE_FILENAME).exists()
            else None,
            "frozen_review_rebase_sha256": _sha256(
                review_manifest_path.parent / FROZEN_REVIEW_REBASE_FILENAME
            )
            if (review_manifest_path.parent / FROZEN_REVIEW_REBASE_FILENAME).exists()
            else None,
        },
        "counts": {
            "reviewed_links": int(len(labels)),
            "effective_labels": int(labels["effective_requires_correction"].notna().sum()),
            "excluded_labels": int(labels["effective_requires_correction"].isna().sum()),
            "completed_review_cases": int(labels["in_completed_review"].sum()),
            "hybrid_comparison_rows": int(len(comparisons)),
            "ambiguous_comparison_rows": int(len(ambiguous)),
            "ambiguous_links": int(ambiguous[["example_id", "id_link"]].drop_duplicates().shape[0]),
            "followup_links": int(len(followup)),
        },
        "files": {key: str(path) for key, path in paths.items() if key != "manifest"},
        "file_sha256": {key: _sha256(path) for key, path in paths.items() if key != "manifest"},
    }
    _atomic_write_json(manifest, paths["manifest"])
    return {
        "output_dir": str(output_dir),
        "manifest": str(paths["manifest"]),
        "comparison_rows": int(len(comparisons)),
        "ambiguous_links": manifest["counts"]["ambiguous_links"],
        "followup_links": manifest["counts"]["followup_links"],
        "profile": profile.to_dict("records"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare per-link hybrid whole-mean/near-junction continuity rules without changing labels or models."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--endpoint-widths", default=str(DEFAULT_ENDPOINTS_PATH))
    parser.add_argument("--junction-manifest", default=str(DEFAULT_JUNCTION_MANIFEST_PATH))
    parser.add_argument("--adjudication-review", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--adjudication-manifest", default=str(DEFAULT_REVIEW_MANIFEST_PATH))
    parser.add_argument("--window-id", default=DEFAULT_WINDOW_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_hybrid_junction_audit(
        dataset_path=args.dataset,
        endpoints_path=args.endpoint_widths,
        junction_manifest_path=args.junction_manifest,
        review_path=args.adjudication_review,
        review_manifest_path=args.adjudication_manifest,
        output_dir=args.output_dir,
        window_id=args.window_id,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
