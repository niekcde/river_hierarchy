from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class RapidStateContext:
    state_id: str
    state_dir: Path
    directed_links_path: Path
    directed_nodes_path: Path
    link_widths_path: Path
    rapid_prep_dir: Path
    rapid_run_dir: Path
    status: str
    depth: int
    state_role: str


def load_state_registry(experiment_dir: str | Path) -> pd.DataFrame:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry_path = experiment_path / "state_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"State registry was not found: {registry_path}")
    frame = pd.read_csv(registry_path)
    if "state_id" not in frame.columns or "state_dir" not in frame.columns:
        raise ValueError(f"State registry {registry_path} is missing required columns.")
    return frame


def _optional_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _resolve_link_widths_path(row: pd.Series) -> Path:
    variant_output_dir = _optional_path(row.get("variant_output_dir"))
    if variant_output_dir is not None:
        candidate = variant_output_dir / "widths" / "links_with_width_families.gpkg"
        if candidate.exists():
            return candidate
    directed_links_path = _optional_path(row.get("directed_links_path"))
    if directed_links_path is None:
        raise ValueError(f"State {row.get('state_id')} is missing directed_links_path.")
    return directed_links_path


def iter_preparable_states(
    registry: pd.DataFrame,
    *,
    include_base_state: bool = True,
) -> Iterable[RapidStateContext]:
    for _, row in registry.iterrows():
        state_role = str(row.get("state_role", "") or "")
        if not include_base_state and state_role == "base":
            continue
        status = str(row.get("status", "") or "")
        if status and status != "complete":
            continue

        state_dir = _optional_path(row.get("state_dir"))
        directed_links_path = _optional_path(row.get("directed_links_path"))
        directed_nodes_path = _optional_path(row.get("directed_nodes_path"))
        if state_dir is None or directed_links_path is None or directed_nodes_path is None:
            continue
        if not directed_links_path.exists() or not directed_nodes_path.exists():
            continue

        yield RapidStateContext(
            state_id=str(row["state_id"]),
            state_dir=state_dir,
            directed_links_path=directed_links_path,
            directed_nodes_path=directed_nodes_path,
            link_widths_path=_resolve_link_widths_path(row),
            rapid_prep_dir=state_dir / "rapid" / "prep",
            rapid_run_dir=state_dir / "rapid" / "run",
            status=status or "complete",
            depth=int(row.get("depth", 0) or 0),
            state_role=state_role or "derived",
        )
