"""Explicit control-artifact helpers for sampled and sensitivity runners."""

from __future__ import annotations

import json
from pathlib import Path


def default_single_edge_control_path() -> Path:
    """Return the repository-backed single-edge control artifact path."""
    return Path(__file__).resolve().parents[3] / "configs" / "single_edge_control.json"


def load_single_edge_control(control_path: str | Path | None = None) -> tuple[dict, Path]:
    """Load and validate the explicit single-edge control specification."""
    path = Path(control_path) if control_path is not None else default_single_edge_control_path()
    if not path.exists():
        raise FileNotFoundError(f"Single-edge control artifact not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    required = [
        "control_type",
        "network_id",
        "geometry_id",
        "sample_type",
        "network_type",
        "length_source",
        "width_source",
    ]
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Single-edge control artifact missing keys: {missing}")
    if spec["control_type"] != "single_edge":
        raise ValueError(
            f"Unsupported control_type in {path}: {spec['control_type']!r}. "
            "Expected 'single_edge'."
        )
    if spec["network_type"] != "single_edge":
        raise ValueError(
            f"Unsupported network_type in {path}: {spec['network_type']!r}. "
            "Expected 'single_edge'."
        )

    return spec, path


def _resolve_source_value(source: dict, meta: dict, *, label: str) -> float:
    kind = source.get("kind")
    if kind != "meta_field":
        raise ValueError(f"Unsupported {label} source kind: {kind!r}")

    field = source.get("field")
    if not field:
        raise ValueError(f"Missing {label} source field.")
    if field not in meta:
        raise KeyError(f"Field {field!r} not available in metadata for {label}.")

    value = float(meta[field])
    if value <= 0:
        raise ValueError(f"Resolved {label} must be > 0. Got {value} from field {field!r}.")
    return value


def resolve_single_edge_control(spec: dict, meta: dict) -> dict:
    """Resolve the explicit control spec against experiment metadata."""
    meta = dict(meta)
    resolved = {
        "control_type": "single_edge",
        "network_id": int(spec["network_id"]),
        "geometry_id": int(spec["geometry_id"]),
        "sample_type": str(spec["sample_type"]),
        "network_type": str(spec["network_type"]),
        "length_m": _resolve_source_value(spec["length_source"], meta, label="length"),
        "width_m": _resolve_source_value(spec["width_source"], meta, label="width"),
    }
    return resolved


__all__ = [
    "default_single_edge_control_path",
    "load_single_edge_control",
    "resolve_single_edge_control",
]
