from gauge_sword_match.hydraulics import (
    classify_kinematic_candidate,
    compute_froude,
    compute_reference_area,
    compute_reference_depth,
    compute_reference_velocity,
    compute_tplus,
)


def test_basic_hydraulic_metric_formulas():
    depth = compute_reference_depth(100.0, 20.0)
    area = compute_reference_area(100.0, depth)
    velocity = compute_reference_velocity(1_000.0, area)
    froude = compute_froude(velocity, depth)
    tplus = compute_tplus(48.0, velocity, 0.002, depth)

    assert depth == 5.0
    assert area == 500.0
    assert velocity == 2.0
    assert round(froude, 4) == round((2.0**2) / (9.80665 * 5.0), 4)
    assert round(tplus, 2) == 138.24


def test_classify_kinematic_candidate_uses_threshold_curve():
    assert classify_kinematic_candidate(0.2, 120.0, regime_tplus_min=80, regime_froude_t0=0.9, regime_tplus_end=1000, regime_froude_end=0.9) is True
    assert classify_kinematic_candidate(1.2, 120.0, regime_tplus_min=80, regime_froude_t0=0.9, regime_tplus_end=1000, regime_froude_end=0.9) is False
    assert classify_kinematic_candidate(0.2, 20.0, regime_tplus_min=80, regime_froude_t0=0.9, regime_tplus_end=1000, regime_froude_end=0.9) is False
