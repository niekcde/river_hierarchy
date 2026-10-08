"""Explicit, versioned settings accepted after the October 2026 hydraulic audit.

Legacy dataclass defaults remain available for reproducing historical runs.
This policy requires stored BASED anchors and reconstructed shared-head contexts.
"""
from .prep import RapidPrepConfig

POLICY_VERSION = "shared-head-anchored-depth-signed-inflow-20261003"


def production_prep_config() -> RapidPrepConfig:
    return RapidPrepConfig(
        kb_mode="based_reference_section",
        depth_width_exponent=0.65,
        n_manning=0.035,
        x_value=0.1,
        target_subreach_length_m=200.0,
        min_effective_length_for_k_m=200.0,
        use_celerity_capping=True,
        min_celerity_mps=0.05,
        max_celerity_mps=5.0,
        coefficient_policy="signed_current_inflow",
        initial_flow_mode="steady_first_inflow",
        use_stability_effective_length_cap=False,
        use_routing_celerity_regularization=False,
    )
