from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class KValueConfig:
    width_field: str = "wid_adj_routing"
    x_value: float = 0.1
    kb_value: float = 20.0
    kb_source_method: str = "fixed_scalar"
    kb_reference_discharge_cms: float | None = None
    kb_reference_width_m: float | None = None
    kb_reference_slope: float | None = None
    kb_reference_depth_m: float | None = None
    n_manning: float = 0.35
    min_width: float = 1.0
    min_effective_length_m: float | None = None
    use_celerity_capping: bool = False
    min_celerity_mps: float = 0.28
    max_celerity_mps: float = 1.524


WIDTH_FIELD_CANDIDATES = (
    "wid_adj_routing",
    "wid_adj_wet",
    "wid_adj",
    "wid_wet",
    "wid",
)


def resolve_width_field(
    columns: pd.Index,
    preferred_field: str,
) -> str:
    if preferred_field in columns:
        return preferred_field
    for column in WIDTH_FIELD_CANDIDATES:
        if column in columns:
            return column
    raise ValueError(
        "Could not resolve a width field for RAPID K-value preparation. "
        f"Preferred field '{preferred_field}' was not found."
    )


def compute_k_values(
    link_frame: pd.DataFrame,
    *,
    config: KValueConfig | None = None,
) -> pd.DataFrame:
    config = config or KValueConfig()
    if config.n_manning <= 0:
        raise ValueError("RAPID K-value preparation requires a positive Manning roughness.")
    if config.kb_value <= 0:
        raise ValueError("RAPID K-value preparation requires a positive kb value.")
    if config.min_width <= 0:
        raise ValueError("RAPID K-value preparation requires a positive minimum width.")
    if config.min_effective_length_m is not None and config.min_effective_length_m <= 0:
        raise ValueError("RAPID effective-length floor must be positive when provided.")
    if config.min_celerity_mps <= 0 or config.max_celerity_mps <= 0:
        raise ValueError("RAPID celerity caps must be positive when provided.")
    if config.min_celerity_mps > config.max_celerity_mps:
        raise ValueError("RAPID celerity cap minimum cannot exceed the maximum.")

    width_field = resolve_width_field(link_frame.columns, config.width_field)

    frame = link_frame.copy()
    frame["rapid_width_source_field"] = width_field
    frame["rapid_width_m_raw"] = pd.to_numeric(frame[width_field], errors="coerce")
    frame["rapid_width_adjusted"] = frame["rapid_width_m_raw"].isna() | frame["rapid_width_m_raw"].le(0)
    frame["rapid_width_m"] = frame["rapid_width_m_raw"].where(
        ~frame["rapid_width_adjusted"],
        config.min_width,
    )
    frame["rapid_width_m"] = frame["rapid_width_m"].clip(lower=config.min_width).astype(float)

    frame["rapid_x"] = float(config.x_value)
    frame["rapid_kb_value"] = float(config.kb_value)
    frame["rapid_kb_source_method"] = str(config.kb_source_method)
    frame["rapid_kb_reference_discharge_cms"] = (
        float(config.kb_reference_discharge_cms) if config.kb_reference_discharge_cms is not None else np.nan
    )
    frame["rapid_kb_reference_width_m"] = (
        float(config.kb_reference_width_m) if config.kb_reference_width_m is not None else np.nan
    )
    frame["rapid_kb_reference_slope"] = (
        float(config.kb_reference_slope) if config.kb_reference_slope is not None else np.nan
    )
    frame["rapid_kb_reference_depth_m"] = (
        float(config.kb_reference_depth_m) if config.kb_reference_depth_m is not None else np.nan
    )
    frame["rapid_celerity_mps_raw"] = (
        (5.0 / (3.0 * float(config.n_manning)))
        * np.sqrt(frame["slope_used"].astype(float))
        * (
            (frame["rapid_width_m"].astype(float) ** (2.0 / 3.0))
            / (float(config.kb_value) ** (2.0 / 3.0))
        )
    ).astype(float)
    if config.use_celerity_capping:
        frame["rapid_celerity_mps"] = frame["rapid_celerity_mps_raw"].clip(
            lower=float(config.min_celerity_mps),
            upper=float(config.max_celerity_mps),
        )
        frame["rapid_celerity_capped"] = ~np.isclose(
            frame["rapid_celerity_mps"],
            frame["rapid_celerity_mps_raw"],
            rtol=1e-12,
            atol=1e-12,
        )
        frame["rapid_k_source_method"] = np.where(
            frame["rapid_celerity_capped"],
            "hydraulic_celerity_capped",
            "hydraulic_celerity_uncapped",
        )
    else:
        frame["rapid_celerity_mps"] = frame["rapid_celerity_mps_raw"].astype(float)
        frame["rapid_celerity_capped"] = False
        frame["rapid_k_source_method"] = "hydraulic_celerity_uncapped"

    frame["rapid_celerity_cap_enabled"] = bool(config.use_celerity_capping)
    frame["rapid_celerity_cap_min_mps"] = float(config.min_celerity_mps)
    frame["rapid_celerity_cap_max_mps"] = float(config.max_celerity_mps)
    frame["rapid_effective_length_floor_m"] = (
        float(config.min_effective_length_m) if config.min_effective_length_m is not None else np.nan
    )
    if config.min_effective_length_m is not None:
        frame["rapid_effective_length_m"] = frame["link_length_m"].astype(float).clip(
            lower=float(config.min_effective_length_m)
        )
    else:
        frame["rapid_effective_length_m"] = frame["link_length_m"].astype(float)
    frame["rapid_length_floor_applied"] = ~np.isclose(
        frame["rapid_effective_length_m"],
        frame["link_length_m"].astype(float),
        rtol=1e-12,
        atol=1e-12,
    )
    frame["rapid_k"] = (frame["rapid_effective_length_m"].astype(float) / frame["rapid_celerity_mps"]).astype(float)
    return frame


def compute_dt_bounds(
    k_values: pd.Series,
    *,
    x_value: float,
) -> tuple[float, float]:
    finite = pd.to_numeric(k_values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        raise ValueError("No finite RAPID K values were available for routing dt selection.")
    dt_min = float(np.max(2.0 * finite * x_value))
    dt_max = float(np.min(2.0 * finite * (1.0 - x_value)))
    return dt_min, dt_max


def apply_stability_effective_length_cap(
    link_frame: pd.DataFrame,
    *,
    x_value: float,
    min_celerity_mps: float,
    safety_margin_seconds: float,
    celerity_rel_buffer: float = 0.03,
) -> pd.DataFrame:
    if x_value <= 0:
        raise ValueError("Muskingum X must be positive when applying the stability length cap.")
    if min_celerity_mps <= 0:
        raise ValueError("Minimum celerity must be positive when applying the stability length cap.")
    if safety_margin_seconds < 0:
        raise ValueError("Stability length-cap safety margin must be nonnegative.")
    if celerity_rel_buffer < 0:
        raise ValueError("Celerity relative buffer must be nonnegative.")

    frame = link_frame.copy()
    current_effective_length = pd.to_numeric(frame["rapid_effective_length_m"], errors="coerce").astype(float)
    floor_length = pd.to_numeric(frame.get("rapid_effective_length_floor_m"), errors="coerce")
    floor_length = floor_length.fillna(-np.inf).astype(float)

    frame["rapid_stability_length_cap_enabled"] = True
    frame["rapid_stability_length_cap_margin_seconds"] = float(safety_margin_seconds)
    frame["rapid_effective_length_m_pre_stability_cap"] = current_effective_length
    frame["rapid_stability_length_cap_candidate"] = False
    frame["rapid_stability_length_cap_applied"] = False
    frame["rapid_stability_length_cap_triggered"] = False
    frame["rapid_stability_length_cap_target_m"] = np.nan
    frame["rapid_stability_length_cap_active_m"] = np.nan

    dt_min_before, dt_max_before = compute_dt_bounds(frame["rapid_k"], x_value=x_value)
    frame["rapid_stability_length_cap_dt_min_before_seconds"] = float(dt_min_before)
    frame["rapid_stability_length_cap_dt_max_before_seconds"] = float(dt_max_before)

    target_upper_dt = dt_max_before - float(safety_margin_seconds)
    if dt_min_before <= target_upper_dt + 1e-9 or target_upper_dt <= 0:
        frame["rapid_stability_length_cap_dt_min_after_seconds"] = float(dt_min_before)
        frame["rapid_stability_length_cap_dt_max_after_seconds"] = float(dt_max_before)
        return frame

    celerity = pd.to_numeric(frame["rapid_celerity_mps"], errors="coerce").astype(float)
    near_floor_threshold = float(min_celerity_mps) * (1.0 + float(celerity_rel_buffer))
    target_cap = (target_upper_dt * celerity / (2.0 * x_value)).astype(float)
    if not np.isfinite(target_cap).any():
        frame["rapid_stability_length_cap_dt_min_after_seconds"] = float(dt_min_before)
        frame["rapid_stability_length_cap_dt_max_after_seconds"] = float(dt_max_before)
        return frame

    active_cap = np.maximum(target_cap, floor_length)
    candidate_mask = (
        celerity.le(near_floor_threshold)
        & current_effective_length.gt(active_cap)
        & np.isfinite(active_cap)
    )

    frame["rapid_stability_length_cap_triggered"] = True
    frame["rapid_stability_length_cap_target_m"] = target_cap.astype(float)
    frame["rapid_stability_length_cap_active_m"] = active_cap.astype(float)
    frame["rapid_stability_length_cap_candidate"] = candidate_mask

    if candidate_mask.any():
        updated_effective_length = current_effective_length.where(~candidate_mask, active_cap)
        frame["rapid_effective_length_m"] = updated_effective_length.astype(float)
        frame["rapid_stability_length_cap_applied"] = candidate_mask
        frame["rapid_k"] = (frame["rapid_effective_length_m"].astype(float) / frame["rapid_celerity_mps"]).astype(float)
        frame["rapid_k_source_method"] = np.where(
            frame["rapid_stability_length_cap_applied"],
            frame["rapid_k_source_method"].astype(str) + "+stability_length_cap",
            frame["rapid_k_source_method"],
        )

    dt_min_after, dt_max_after = compute_dt_bounds(frame["rapid_k"], x_value=x_value)
    frame["rapid_stability_length_cap_dt_min_after_seconds"] = float(dt_min_after)
    frame["rapid_stability_length_cap_dt_max_after_seconds"] = float(dt_max_after)
    return frame


def compute_routing_dt_seconds(
    k_values: pd.Series,
    *,
    x_value: float,
    forcing_dt_seconds: int,
    preferred_routing_dt_seconds: int | None = None,
) -> int:
    dt_min, dt_max = compute_dt_bounds(k_values, x_value=x_value)
    return compute_routing_dt_seconds_from_bounds(
        dt_min=dt_min,
        dt_max=dt_max,
        forcing_dt_seconds=forcing_dt_seconds,
        preferred_routing_dt_seconds=preferred_routing_dt_seconds,
        x_value=x_value,
    )


def compute_routing_dt_seconds_from_bounds(
    *,
    dt_min: float,
    dt_max: float,
    forcing_dt_seconds: int,
    preferred_routing_dt_seconds: int | None = None,
    x_value: float | None = None,
) -> int:
    if dt_min > dt_max:
        x_text = f"{x_value}" if x_value is not None else "unknown"
        raise ValueError(
            f"No stable RAPID routing timestep exists for x={x_text}. "
            f"Computed dt_min={dt_min:.3f} > dt_max={dt_max:.3f}."
        )

    candidates: list[int] = []
    for candidate in range(1, int(forcing_dt_seconds) + 1):
        if forcing_dt_seconds % candidate != 0:
            continue
        if candidate + 1e-9 < dt_min or candidate - 1e-9 > dt_max:
            continue
        candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "Could not find a routing timestep that both satisfies the Muskingum "
            f"stability bounds [{dt_min:.3f}, {dt_max:.3f}] and evenly divides "
            f"the forcing timestep {forcing_dt_seconds}."
        )

    target = float(preferred_routing_dt_seconds) if preferred_routing_dt_seconds is not None else 0.5 * (dt_min + dt_max)
    best = min(candidates, key=lambda value: (abs(value - target), -value))
    return int(best)


def find_nearby_compatible_forcing_and_routing_dt(
    k_values: pd.Series,
    *,
    x_value: float,
    forcing_dt_seconds: int,
    preferred_routing_dt_seconds: int | None = None,
    search_max_offset_seconds: int = 120,
) -> tuple[int, int]:
    dt_min, dt_max = compute_dt_bounds(k_values, x_value=x_value)
    if dt_min > dt_max:
        raise ValueError(
            f"No stable RAPID routing timestep exists for x={x_value}. "
            f"Computed dt_min={dt_min:.3f} > dt_max={dt_max:.3f}."
        )

    preferred = int(preferred_routing_dt_seconds) if preferred_routing_dt_seconds is not None else int(round(0.5 * (dt_min + dt_max)))
    multiples_of_five: list[tuple[int, int]] = []
    integer_candidates: list[tuple[int, int]] = []
    for offset in range(0, int(search_max_offset_seconds) + 1):
        forcing_candidates = [forcing_dt_seconds] if offset == 0 else [forcing_dt_seconds - offset, forcing_dt_seconds + offset]
        for candidate_forcing_dt in forcing_candidates:
            if candidate_forcing_dt <= 0:
                continue
            try:
                candidate_routing_dt = compute_routing_dt_seconds_from_bounds(
                    dt_min=dt_min,
                    dt_max=dt_max,
                    forcing_dt_seconds=int(candidate_forcing_dt),
                    preferred_routing_dt_seconds=preferred,
                    x_value=x_value,
                )
            except ValueError:
                continue
            record = (int(candidate_forcing_dt), int(candidate_routing_dt))
            if int(candidate_forcing_dt) % 5 == 0 and int(candidate_routing_dt) % 5 == 0:
                multiples_of_five.append(record)
            integer_candidates.append(record)

    if multiples_of_five:
        return min(
            multiples_of_five,
            key=lambda pair: (abs(pair[1] - preferred), abs(pair[0] - forcing_dt_seconds), pair[0], -pair[1]),
        )
    if integer_candidates:
        return min(
            integer_candidates,
            key=lambda pair: (abs(pair[1] - preferred), abs(pair[0] - forcing_dt_seconds), pair[0], -pair[1]),
        )
    raise ValueError(
        "Could not find a nearby forcing timestep with any routing timestep that satisfies the Muskingum "
        f"stability bounds [{dt_min:.3f}, {dt_max:.3f}] and evenly divides the forcing timestep."
    )


def _stable_integer_candidates(
    dt_min: float,
    dt_max: float,
    *,
    multiple_seconds: int,
) -> list[int]:
    """Return stable integer timesteps, preferring a requested regular grid."""
    if not np.isfinite(dt_min) or not np.isfinite(dt_max) or dt_min > dt_max:
        return []
    multiple = max(1, int(multiple_seconds))
    first = int(np.ceil(float(dt_min) / multiple)) * multiple
    last = int(np.floor(float(dt_max) / multiple)) * multiple
    if first <= last:
        return list(range(first, last + 1, multiple))

    # Very narrow valid intervals may contain no multiple of the requested grid.
    # Preserve stability with an integer-second fallback rather than rounding out
    # of bounds.
    first_integer = int(np.ceil(float(dt_min)))
    last_integer = int(np.floor(float(dt_max)))
    if first_integer <= last_integer:
        return list(range(first_integer, last_integer + 1))
    return []


def select_coordinated_routing_timesteps(
    state_bounds: Mapping[str, tuple[float, float]],
    *,
    preferred_routing_dt_seconds: int | None = None,
    multiple_seconds: int = 5,
) -> dict[str, object]:
    """Select stable routing timesteps jointly for every experiment state.

    A single multiple-of-five timestep is used when the state stability
    intervals overlap. When they do not, a common central target is projected
    into each interval so that the selected stable values remain as close as
    the intervals permit.
    """
    if not state_bounds:
        raise ValueError("Cannot coordinate routing timesteps without state stability bounds.")

    normalized: dict[str, tuple[float, float]] = {}
    state_candidates: dict[str, list[int]] = {}
    for state_id, raw_bounds in state_bounds.items():
        dt_min, dt_max = (float(raw_bounds[0]), float(raw_bounds[1]))
        if not np.isfinite(dt_min) or not np.isfinite(dt_max):
            raise ValueError(f"State {state_id} has non-finite routing timestep bounds.")
        if dt_min > dt_max:
            raise ValueError(
                f"State {state_id} has no stable routing timestep: "
                f"dt_min={dt_min:.3f} > dt_max={dt_max:.3f}."
            )
        candidates = _stable_integer_candidates(
            dt_min,
            dt_max,
            multiple_seconds=multiple_seconds,
        )
        if not candidates:
            raise ValueError(
                f"State {state_id} has no stable integer-second routing timestep "
                f"inside [{dt_min:.3f}, {dt_max:.3f}]."
            )
        normalized[str(state_id)] = (dt_min, dt_max)
        state_candidates[str(state_id)] = candidates

    common_min = max(bounds[0] for bounds in normalized.values())
    common_max = min(bounds[1] for bounds in normalized.values())
    common_candidates = _stable_integer_candidates(
        common_min,
        common_max,
        multiple_seconds=multiple_seconds,
    )
    preferred = (
        float(preferred_routing_dt_seconds)
        if preferred_routing_dt_seconds is not None
        else None
    )

    if common_candidates:
        if preferred is not None and preferred in common_candidates:
            target = preferred
        else:
            target = 0.5 * (common_min + common_max)
        selected_common = min(
            common_candidates,
            key=lambda value: (abs(float(value) - target), -value),
        )
        selections = {state_id: int(selected_common) for state_id in normalized}
        mode = "common_intersection"
        central_target = float(selected_common)
    else:
        # The midpoint between the highest lower bound and lowest upper bound is
        # the minimax centre of a disjoint set of one-dimensional intervals.
        # Projecting it into every interval yields the smallest possible spread
        # before regular-grid rounding.
        central_target = 0.5 * (common_min + common_max)
        selections = {
            state_id: int(
                min(
                    candidates,
                    key=lambda value: (
                        abs(float(value) - central_target),
                        abs(float(value) - preferred) if preferred is not None else 0.0,
                        -value,
                    ),
                )
            )
            for state_id, candidates in state_candidates.items()
        }
        mode = "coordinated_state_specific"

    state_rows = {
        state_id: {
            "dt_min_seconds": float(normalized[state_id][0]),
            "dt_max_seconds": float(normalized[state_id][1]),
            "routing_dt_seconds": int(selections[state_id]),
            # A one-to-one forcing/routing interval is the smallest exact
            # alignment and avoids independently changing the two clocks.
            "forcing_dt_seconds": int(selections[state_id]),
            "selection_source": mode,
            "used_multiple_seconds": int(multiple_seconds),
            "multiple_fallback_used": bool(
                int(selections[state_id]) % max(1, int(multiple_seconds)) != 0
            ),
        }
        for state_id in normalized
    }
    return {
        "selection_mode": mode,
        "preferred_routing_dt_seconds": (
            int(preferred_routing_dt_seconds)
            if preferred_routing_dt_seconds is not None
            else None
        ),
        "multiple_seconds": int(multiple_seconds),
        "common_dt_min_seconds": float(common_min),
        "common_dt_max_seconds": float(common_max),
        "common_intersection_exists": bool(common_candidates),
        "central_target_seconds": float(central_target),
        "selected_unique_routing_dt_seconds": sorted(set(selections.values())),
        "states": state_rows,
    }
