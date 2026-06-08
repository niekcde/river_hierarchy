from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .collapse_experiment import CollapseExperimentOutputs, run_collapse_experiment
    from .reference_context import ResolvedKbContext, resolve_kb_context
    from .variant_generation import (
        NetworkVariantOutputs,
        compute_width_families,
        generate_network_variant,
        resolve_selected_unit_ids,
    )


__all__ = [
    "CollapseExperimentOutputs",
    "NetworkVariantOutputs",
    "ResolvedKbContext",
    "compute_width_families",
    "generate_network_variant",
    "resolve_kb_context",
    "run_collapse_experiment",
    "resolve_selected_unit_ids",
]


def __getattr__(name: str) -> Any:
    if name in {"CollapseExperimentOutputs", "run_collapse_experiment"}:
        from .collapse_experiment import CollapseExperimentOutputs, run_collapse_experiment

        exports = {
            "CollapseExperimentOutputs": CollapseExperimentOutputs,
            "run_collapse_experiment": run_collapse_experiment,
        }
        return exports[name]
    if name in {"ResolvedKbContext", "resolve_kb_context"}:
        from .reference_context import ResolvedKbContext, resolve_kb_context

        exports = {
            "ResolvedKbContext": ResolvedKbContext,
            "resolve_kb_context": resolve_kb_context,
        }
        return exports[name]
    if name in {
        "NetworkVariantOutputs",
        "compute_width_families",
        "generate_network_variant",
        "resolve_selected_unit_ids",
    }:
        from .variant_generation import (
            NetworkVariantOutputs,
            compute_width_families,
            generate_network_variant,
            resolve_selected_unit_ids,
        )

        exports = {
            "NetworkVariantOutputs": NetworkVariantOutputs,
            "compute_width_families": compute_width_families,
            "generate_network_variant": generate_network_variant,
            "resolve_selected_unit_ids": resolve_selected_unit_ids,
        }
        return exports[name]
    raise AttributeError(name)
