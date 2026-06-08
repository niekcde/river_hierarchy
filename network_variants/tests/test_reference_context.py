from __future__ import annotations

import pytest

from network_variants.reference_context import resolve_kb_context


def test_resolve_kb_context_fixed_scalar() -> None:
    resolved = resolve_kb_context(kb_mode="fixed", kb_value=18.5)

    assert resolved.kb_mode == "fixed"
    assert resolved.kb_value == 18.5
    assert resolved.kb_source_method == "fixed_scalar"
    assert resolved.reference_discharge_cms is None
    assert resolved.reference_width_m is None
    assert resolved.reference_slope is None
    assert resolved.reference_depth_m is None
    assert resolved.reference_section_summary_path == ""
    assert resolved.based_model_path == ""


def test_resolve_kb_context_based_reference_section() -> None:
    resolved = resolve_kb_context(
        kb_mode="based_reference_section",
        reference_kb_summary={
            "kb_value": 12.5,
            "kb_source_method": "based_reference_section",
            "reference_discharge_cms": 120.0,
            "reference_width_m": 320.0,
            "reference_slope": 1.2e-4,
            "reference_depth_m": 25.6,
            "reference_section_summary_path": "/tmp/reference_section_summary.json",
            "based_model_path": "/tmp/based_model_v2.ubj",
        },
    )

    assert resolved.kb_mode == "based_reference_section"
    assert resolved.kb_value == 12.5
    assert resolved.kb_source_method == "based_reference_section"
    assert resolved.reference_discharge_cms == 120.0
    assert resolved.reference_width_m == 320.0
    assert resolved.reference_slope == 1.2e-4
    assert resolved.reference_depth_m == 25.6
    assert resolved.reference_section_summary_path == "/tmp/reference_section_summary.json"
    assert resolved.based_model_path == "/tmp/based_model_v2.ubj"


def test_resolve_kb_context_rejects_missing_reference_summary() -> None:
    with pytest.raises(ValueError, match="reference_kb_summary"):
        resolve_kb_context(kb_mode="based_reference_section")
