from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class KValueConfig:
    width_field: str = "wid_adj_wet"
    x_value: float = 0.1
    kb_value: float = 20.0
    n_manning: float = 0.35
    min_width: float = 1.0


WIDTH_FIELD_CANDIDATES = (
    "wid_adj_wet",
    "wid_adj_total",
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
    frame["rapid_k"] = (
        (3.0 / 5.0)
        * float(config.n_manning)
        * (
            (frame["link_length_m"] / np.sqrt(frame["slope_used"]))
            * ((float(config.kb_value) ** (2.0 / 3.0)) / (frame["rapid_width_m"] ** (2.0 / 3.0)))
        )
    ).astype(float)
    return frame


def compute_routing_dt_seconds(
    k_values: pd.Series,
    *,
    x_value: float,
    forcing_dt_seconds: int,
) -> int:
    finite = pd.to_numeric(k_values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        raise ValueError("No finite RAPID K values were available for routing dt selection.")

    dt_min = float(np.max(2.0 * finite * x_value))
    dt_max = float(np.min(2.0 * finite * (1.0 - x_value)))
    if dt_min > dt_max:
        raise ValueError(
            f"No stable RAPID routing timestep exists for x={x_value}. "
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

    midpoint = 0.5 * (dt_min + dt_max)
    best = min(candidates, key=lambda value: (abs(value - midpoint), -value))
    return int(best)
