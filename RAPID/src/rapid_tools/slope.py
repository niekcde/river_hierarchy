from __future__ import annotations

"""Compatibility exports for network hydraulic-context derivation.

This module remains as a stable import path for RAPID preparation code, but the
implementation now lives in `network_variants.hydraulic_context` so the state
hydraulic context can be generated earlier in the variant workflow.
"""

from network_variants.hydraulic_context import (
    SlopeConfig,
    compute_link_slopes,
    compute_section_slope_reference,
)

__all__ = [
    "SlopeConfig",
    "compute_link_slopes",
    "compute_section_slope_reference",
]
