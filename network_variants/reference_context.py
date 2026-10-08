from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ResolvedKbContext:
    """Resolved width-depth-ratio context shared across workflows."""

    kb_mode: str
    kb_value: float
    kb_source_method: str
    reference_discharge_cms: float | None = None
    reference_width_m: float | None = None
    reference_slope: float | None = None
    reference_depth_m: float | None = None
    reference_section_summary_path: str = ""
    based_model_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_required_float(value: object, *, field_name: str, positive: bool = False) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not resolve a finite numeric value for '{field_name}'.")
    numeric = float(numeric)
    if positive and numeric <= 0:
        raise ValueError(f"'{field_name}' must be positive.")
    return numeric


def _coerce_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"Could not resolve a finite numeric value for '{field_name}'.")
    numeric = float(numeric)
    return None if not np.isfinite(numeric) else numeric


def resolve_kb_context(
    *,
    kb_mode: str,
    kb_value: float | None = None,
    reference_kb_summary: Mapping[str, Any] | None = None,
) -> ResolvedKbContext:
    """Resolve the state-level width-depth ratio and its provenance.

    This helper is intentionally workflow-agnostic so both RAPID prep and
    geometry-informed equivalent-resistance metrics can consume the same
    resolved kb context without sharing pipeline-specific code.
    """

    normalized_mode = str(kb_mode or "fixed").strip()
    if normalized_mode == "fixed":
        return ResolvedKbContext(
            kb_mode=normalized_mode,
            kb_value=_coerce_required_float(kb_value, field_name="kb_value", positive=True),
            kb_source_method="fixed_scalar",
        )

    if normalized_mode == "based_reference_section":
        if reference_kb_summary is None:
            raise ValueError("based_reference_section kb mode requires a reference_kb_summary.")
        missing = sorted(
            {
                "kb_value",
                "kb_source_method",
                "reference_discharge_cms",
                "reference_width_m",
                "reference_slope",
                "reference_depth_m",
            }.difference(reference_kb_summary.keys())
        )
        if missing:
            raise ValueError(
                "reference_kb_summary is missing required fields for based_reference_section: "
                + ", ".join(missing)
            )
        return ResolvedKbContext(
            kb_mode=normalized_mode,
            kb_value=_coerce_required_float(reference_kb_summary["kb_value"], field_name="kb_value", positive=True),
            kb_source_method=str(reference_kb_summary["kb_source_method"]),
            reference_discharge_cms=_coerce_optional_float(
                reference_kb_summary.get("reference_discharge_cms"),
                field_name="reference_discharge_cms",
            ),
            reference_width_m=_coerce_optional_float(
                reference_kb_summary.get("reference_width_m"),
                field_name="reference_width_m",
            ),
            reference_slope=_coerce_optional_float(
                reference_kb_summary.get("reference_slope"),
                field_name="reference_slope",
            ),
            reference_depth_m=_coerce_optional_float(
                reference_kb_summary.get("reference_depth_m"),
                field_name="reference_depth_m",
            ),
            reference_section_summary_path=str(reference_kb_summary.get("reference_section_summary_path", "") or ""),
            based_model_path=str(reference_kb_summary.get("based_model_path", "") or ""),
        )

    raise ValueError(
        f"Unsupported kb_mode '{normalized_mode}'. Expected one of: fixed, based_reference_section."
    )


__all__ = ["ResolvedKbContext", "resolve_kb_context"]


def anchored_depth_and_kb(widths, *, reference_width_m, reference_depth_m, exponent):
    """Shared depth geometry for routing and equivalent resistance."""
    width = np.asarray(widths, dtype=float)
    if not np.isfinite(width).all() or (width <= 0).any():
        raise ValueError('Anchored depth requires finite positive widths.')
    values = [reference_width_m, reference_depth_m, exponent]
    if any(v is None or not np.isfinite(v) for v in values):
        raise ValueError('Anchored depth requires finite reference width/depth and exponent.')
    if reference_width_m <= 0 or reference_depth_m <= 0 or not 0 <= exponent <= 1:
        raise ValueError('Invalid anchored depth parameters.')
    depth = reference_depth_m * (width / reference_width_m) ** exponent
    return depth, width / depth
