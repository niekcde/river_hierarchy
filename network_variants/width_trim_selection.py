from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TRIM_SELECTION_VERSION = "width-trim-selection-v3"
SELECTED_SAMPLE_FIELD = "used_in_selected_width"
NO_VALID_CORE_METHOD = "untrimmed_no_valid_core"
MINIMUM_FALLBACK_CORE_SAMPLES = 6


@dataclass(frozen=True)
class TrimSelection:
    method: str
    start: int
    stop: int
    valid_core: bool
    fallback_reason: str
    native_candidate_start: int = -1
    native_candidate_stop: int = -1


def _parse_width_sequence(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return pd.to_numeric(pd.Series(list(value)), errors="coerce").to_numpy(float)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return np.asarray([], dtype=float)
    return pd.to_numeric(pd.Series([part.strip() for part in text.split(",")]), errors="coerce").to_numpy(float)


def _distances(samples: pd.DataFrame) -> np.ndarray:
    x = pd.to_numeric(samples["x"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(samples["y"], errors="coerce").to_numpy(float)
    step = np.hypot(np.diff(x), np.diff(y))
    step = np.where(np.isfinite(step), step, 0.0)
    return np.insert(np.cumsum(step), 0, 0.0)


def _candidate_bounds(widths: np.ndarray, distances: np.ndarray) -> tuple[int, int, bool]:
    if len(widths) != len(distances) or len(widths) < 2:
        return 0, len(distances), False
    if not np.isfinite(widths[[0, -1]]).all() or np.any(widths[[0, -1]] <= 0):
        return 0, len(distances), False
    reverse = np.insert(np.cumsum(np.flipud(np.diff(distances))), 0, 0.0)
    start = int(np.argmin(np.abs(distances - widths[0] / 2.0 * 1.1)))
    stop = int(len(distances) - np.argmin(np.abs(reverse - widths[-1] / 2.0 * 1.1)) - 1)
    return start, stop, bool(start < stop)


def _valid_selected_values(samples: pd.DataFrame, start: int, stop: int) -> bool:
    if start < 0 or stop > len(samples) or start >= stop:
        return False
    selected = samples.iloc[start:stop]
    for field in ("width_total", "width_wet"):
        values = pd.to_numeric(selected[field], errors="coerce")
        if values.empty or values.isna().any() or values.le(0).any():
            return False
    return True


def _fallback_support_is_sufficient(distances: np.ndarray, start: int, stop: int) -> bool:
    if stop - start < MINIMUM_FALLBACK_CORE_SAMPLES:
        return False
    selected = distances[start:stop]
    steps = np.diff(distances)
    finite_steps = steps[np.isfinite(steps) & (steps > 0)]
    if len(selected) < 2 or not finite_steps.size:
        return False
    # Raster centerline samples may alternate cardinal and diagonal moves.
    # The smallest positive step is therefore the local pixel scale; using the
    # median would incorrectly reject six contiguous samples on a bend.
    minimum_span = (MINIMUM_FALLBACK_CORE_SAMPLES - 1) * float(np.min(finite_steps))
    return bool(float(selected[-1] - selected[0]) >= minimum_span - 1e-9)


def select_width_trim(row: pd.Series, samples: pd.DataFrame) -> TrimSelection:
    distances = _distances(samples)
    total = pd.to_numeric(samples["width_total"], errors="coerce").to_numpy(float)
    start, stop, valid = _candidate_bounds(total, distances)
    if valid and _valid_selected_values(samples, start, stop):
        return TrimSelection("two_bank_interior", start, stop, True, "")

    native = _parse_width_sequence(row.get("wid_pix"))
    native_source = "wid_pix"
    if len(native) != len(samples) and "dt_width_global" in samples.columns:
        native = pd.to_numeric(samples["dt_width_global"], errors="coerce").to_numpy(float)
        native_source = "dt_width_global"
    native_start, native_stop, native_valid = _candidate_bounds(native, distances)
    native_values_valid = native_valid and _valid_selected_values(
        samples, native_start, native_stop
    )
    native_support_valid = native_values_valid and _fallback_support_is_sufficient(
        distances, native_start, native_stop
    )
    if native_support_valid:
        return TrimSelection(
            "native_rivgraph_fallback",
            native_start,
            native_stop,
            True,
            f"two_bank_endpoint_trim_invalid;native_source={native_source}",
            native_start,
            native_stop,
        )

    reason = "two_bank_endpoint_trim_invalid"
    if len(native) != len(samples):
        reason += ";native_sample_mapping_unavailable"
    elif native_values_valid:
        reason += (
            ";native_endpoint_trim_insufficient_support"
            f";native_selected_samples={native_stop - native_start}"
            f";minimum_selected_samples={MINIMUM_FALLBACK_CORE_SAMPLES}"
        )
    else:
        reason += ";native_endpoint_trim_invalid"
    return TrimSelection(
        NO_VALID_CORE_METHOD,
        0,
        len(samples),
        False,
        reason,
        native_start if len(native) == len(samples) else -1,
        native_stop if len(native) == len(samples) else -1,
    )


def _stats(values: pd.Series, *, allow_zero: bool = False) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.ge(0) if allow_zero else numeric.gt(0)
    numeric = numeric[np.isfinite(numeric) & valid]
    if numeric.empty:
        return {key: float("nan") for key in ("adj", "med", "p05", "p50", "p95")}
    quantiles = numeric.quantile([0.05, 0.5, 0.95])
    return {
        "adj": float(numeric.mean()),
        "med": float(numeric.median()),
        "p05": float(quantiles.loc[0.05]),
        "p50": float(quantiles.loc[0.5]),
        "p95": float(quantiles.loc[0.95]),
    }


def _optional_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _optional_int(value: Any, default: int = 0) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return int(numeric) if pd.notna(numeric) else int(default)


def apply_authoritative_width_trim_selection(
    links: pd.DataFrame,
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make one trim interval authoritative for summaries, QC, and endpoints."""
    if links.empty or samples.empty:
        return links.copy(), samples.copy(), pd.DataFrame()
    required_samples = {"id_link", "sample_index", "x", "y", "width_total", "width_wet"}
    missing = sorted(required_samples.difference(samples.columns))
    if missing:
        raise ValueError(f"Width samples are missing trim-selection fields: {missing}")

    result = links.copy()
    marked = samples.copy()
    result["id_link"] = pd.to_numeric(result["id_link"], errors="raise").astype(int)
    marked["id_link"] = pd.to_numeric(marked["id_link"], errors="raise").astype(int)
    marked[SELECTED_SAMPLE_FIELD] = False
    marked["selected_width_trim_method"] = ""
    records: list[dict[str, Any]] = []

    link_index = result.set_index("id_link", drop=False)
    for link_id, original_group in marked.groupby("id_link", sort=False):
        if int(link_id) not in link_index.index:
            raise ValueError(f"Width samples reference absent link {int(link_id)}.")
        group = original_group.sort_values("sample_index", kind="mergesort")
        selection = select_width_trim(link_index.loc[int(link_id)], group)
        selected_index = group.index[selection.start:selection.stop]
        marked.loc[group.index, "selected_width_trim_method"] = selection.method
        marked.loc[selected_index, SELECTED_SAMPLE_FIELD] = True
        selected = marked.loc[selected_index]

        record: dict[str, Any] = {
            "id_link": int(link_id),
            "width_trim_selection_version": TRIM_SELECTION_VERSION,
            "width_trim_method": selection.method,
            "width_trim_valid_core": bool(selection.valid_core),
            "width_trim_fallback_reason": selection.fallback_reason,
            "width_trim_start_sample_index": int(group.iloc[selection.start]["sample_index"]),
            "width_trim_stop_sample_index_exclusive": (
                int(group.iloc[selection.stop]["sample_index"])
                if selection.stop < len(group)
                else int(group.iloc[-1]["sample_index"]) + 1
            ),
            "width_trim_sample_count": int(len(selected)),
            "width_trim_native_candidate_start": int(selection.native_candidate_start),
            "width_trim_native_candidate_stop_exclusive": int(selection.native_candidate_stop),
            "width_trim_native_candidate_sample_count": int(
                max(selection.native_candidate_stop - selection.native_candidate_start, 0)
            ),
            "wid_adj_total_two_bank_original": _optional_float(
                link_index.loc[int(link_id)].get(
                    "wid_adj_total_two_bank_original",
                    link_index.loc[int(link_id)].get("wid_adj_total", np.nan),
                )
            ),
            "wid_adj_wet_two_bank_original": _optional_float(
                link_index.loc[int(link_id)].get(
                    "wid_adj_wet_two_bank_original",
                    link_index.loc[int(link_id)].get("wid_adj_wet", np.nan),
                )
            ),
        }
        source_row = link_index.loc[int(link_id)]
        for family, field in (("total", "width_total"), ("wet", "width_wet"), ("dry", "width_dry")):
            if selection.method == NO_VALID_CORE_METHOD:
                for statistic in ("adj", "med", "p05", "p50", "p95"):
                    column = f"wid_{statistic}_{family}"
                    if column in source_row.index:
                        record[column] = _optional_float(source_row[column])
                continue
            if field in selected.columns:
                values = _stats(selected[field], allow_zero=family == "dry")
                for statistic, value in values.items():
                    record[f"wid_{statistic}_{family}"] = value
        total_mean = record.get("wid_adj_total", np.nan)
        wet_mean = record.get("wid_adj_wet", np.nan)
        dry_mean = record.get("wid_adj_dry", np.nan)
        record["wet_fraction_adj"] = (
            _optional_float(source_row.get("wet_fraction_adj", np.nan))
            if selection.method == NO_VALID_CORE_METHOD
            else wet_mean / total_mean if total_mean > 0 else np.nan
        )
        record["dry_fraction_adj"] = (
            _optional_float(source_row.get("dry_fraction_adj", np.nan))
            if selection.method == NO_VALID_CORE_METHOD
            else dry_mean / total_mean if total_mean > 0 else np.nan
        )
        if "n_channels" in selected.columns:
            channels = pd.to_numeric(selected["n_channels"], errors="coerce").dropna()
            record["n_channels_mean"] = (
                _optional_float(source_row.get("n_channels_mean", np.nan))
                if selection.method == NO_VALID_CORE_METHOD
                else float(channels.mean()) if not channels.empty else np.nan
            )
            record["n_channels_max"] = (
                _optional_int(source_row.get("n_channels_max", 0))
                if selection.method == NO_VALID_CORE_METHOD
                else int(channels.max()) if not channels.empty else 0
            )
        if "n_wet_threads" in selected.columns:
            threads = pd.to_numeric(selected["n_wet_threads"], errors="coerce").dropna()
            record["n_wet_threads_mean"] = (
                _optional_float(source_row.get("n_wet_threads_mean", np.nan))
                if selection.method == NO_VALID_CORE_METHOD
                else float(threads.mean()) if not threads.empty else np.nan
            )
            record["n_wet_threads_max"] = (
                _optional_int(source_row.get("n_wet_threads_max", 0))
                if selection.method == NO_VALID_CORE_METHOD
                else int(threads.max()) if not threads.empty else 0
            )
        if "qc_opposition_cos" in selected.columns:
            opposition = pd.to_numeric(selected["qc_opposition_cos"], errors="coerce").dropna()
            record["qc_mean_opposition_cos_trimmed"] = (
                _optional_float(source_row.get("qc_mean_opposition_cos_trimmed", np.nan))
                if selection.method == NO_VALID_CORE_METHOD
                else float(opposition.mean()) if not opposition.empty else np.nan
            )
        records.append(record)

    audit = pd.DataFrame.from_records(records)
    replacement = set(audit.columns).difference({"id_link"})
    result = result.drop(columns=[column for column in replacement if column in result.columns], errors="ignore")
    result = result.merge(audit, on="id_link", how="left", validate="1:1")
    return result, marked, audit


def _parse_link_ids(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_link_ids(item))
        return result
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def resolve_closed_replacement_exclusions(
    child_links: pd.DataFrame,
    parent_links: pd.DataFrame,
    changed_link_ids: list[int] | set[int] | tuple[int, ...],
) -> tuple[list[int], pd.DataFrame]:
    """Exclude only replacements whose parents required no width intervention."""
    changed = {int(value) for value in changed_link_ids}
    parent = parent_links.copy()
    parent["id_link"] = pd.to_numeric(parent["id_link"], errors="raise").astype(int)
    parent = parent.set_index("id_link", drop=False)
    records: list[dict[str, Any]] = []
    excluded: list[int] = []
    candidates = child_links.loc[
        child_links["id_link"].astype(int).isin(changed)
        & child_links.get("lineage_type", pd.Series("", index=child_links.index)).astype(str).eq(
            "collapsed_many_to_one"
        )
    ]
    for row in candidates.itertuples(index=False):
        parent_ids = _parse_link_ids(getattr(row, "matched_parent_link_ids", ""))
        if not parent_ids:
            dominant = pd.to_numeric(
                pd.Series([getattr(row, "dominant_parent_link_id", np.nan)]), errors="coerce"
            ).iloc[0]
            parent_ids = [int(dominant)] if pd.notna(dominant) else []
        available = parent.loc[parent.index.intersection(parent_ids)]
        correction = (
            available.get("width_correction_applied", pd.Series(False, index=available.index))
            .fillna(False)
            .astype(bool)
        )
        trim_method = available.get(
            "width_trim_method", pd.Series("legacy_unknown", index=available.index)
        ).astype(str)
        fallback = ~trim_method.isin({"two_bank_interior", "legacy_unknown"})
        original = pd.to_numeric(
            available.get("wid_adj_wet_two_bank_original", pd.Series(np.nan, index=available.index)),
            errors="coerce",
        )
        selected = pd.to_numeric(
            available.get("wid_adj_wet", pd.Series(np.nan, index=available.index)), errors="coerce"
        )
        material_trim_change = (
            original.notna()
            & selected.notna()
            & ((selected - original).abs() > np.maximum(original.abs() * 1e-9, 1e-9))
        )
        parent_intervention = bool(
            available.empty
            or correction.any()
            or fallback.any()
            or material_trim_change.any()
        )
        if not parent_intervention:
            excluded.append(int(row.id_link))
        records.append(
            {
                "id_link": int(row.id_link),
                "matched_parent_link_ids": ",".join(str(value) for value in parent_ids),
                "available_parent_count": int(len(available)),
                "parent_width_intervention": parent_intervention,
                "closed_replacement_excluded": not parent_intervention,
                "decision": (
                    "evaluate_replacement_parent_width_intervention"
                    if parent_intervention
                    else "retain_raw_closed_two_link_replacement"
                ),
            }
        )
    return sorted(excluded), pd.DataFrame.from_records(records)


def build_parent_child_width_diagnostics(
    child_links: pd.DataFrame,
    parent_links: pd.DataFrame,
) -> pd.DataFrame:
    """Compare collapsed-child routing width with its parents without acting on it.

    These values are deliberately diagnostic only.  They must not influence
    trimming, review classification, or the routing-width solution.
    """
    required = {"id_link", "wid_adj_routing"}
    if child_links.empty or parent_links.empty or not required.issubset(child_links.columns):
        return pd.DataFrame()
    if not required.issubset(parent_links.columns):
        return pd.DataFrame()

    parent = parent_links.copy()
    parent["id_link"] = pd.to_numeric(parent["id_link"], errors="coerce")
    parent["wid_adj_routing"] = pd.to_numeric(parent["wid_adj_routing"], errors="coerce")
    parent = parent.dropna(subset=["id_link"]).copy()
    parent["id_link"] = parent["id_link"].astype(int)
    parent = parent.set_index("id_link", drop=False)
    lineage = child_links.get("lineage_type", pd.Series("", index=child_links.index)).astype(str)
    records: list[dict[str, Any]] = []
    for index, row in child_links.loc[lineage.eq("collapsed_many_to_one")].iterrows():
        parent_ids = _parse_link_ids(row.get("matched_parent_link_ids", ""))
        if not parent_ids:
            dominant = pd.to_numeric(pd.Series([row.get("dominant_parent_link_id")]), errors="coerce").iloc[0]
            parent_ids = [int(dominant)] if pd.notna(dominant) else []
        available_ids = [value for value in parent_ids if value in parent.index]
        parent_widths = pd.to_numeric(
            parent.loc[available_ids, "wid_adj_routing"] if available_ids else pd.Series(dtype=float),
            errors="coerce",
        ).dropna()
        child_width = pd.to_numeric(pd.Series([row.get("wid_adj_routing")]), errors="coerce").iloc[0]
        parent_sum = float(parent_widths.sum()) if len(parent_widths) == len(parent_ids) and parent_ids else np.nan
        difference = float(child_width - parent_sum) if pd.notna(child_width) and np.isfinite(parent_sum) else np.nan
        records.append(
            {
                "id_link": int(row["id_link"]),
                "parent_child_width_diagnostic_only": True,
                "parent_routing_width_link_ids": ",".join(str(value) for value in parent_ids),
                "parent_routing_width_available_count": int(len(parent_widths)),
                "parent_routing_width_sum": parent_sum,
                "child_routing_width": float(child_width) if pd.notna(child_width) else np.nan,
                "parent_child_routing_width_difference": difference,
                "parent_child_routing_width_relative_difference": (
                    difference / parent_sum if np.isfinite(difference) and parent_sum > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(records)
