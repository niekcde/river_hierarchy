"""Helpers for config-driven synthetic workflow entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synthetic_runs.core import Params


def load_json_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data, path


def require_keys(config: dict[str, Any], keys: list[str], *, label: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise KeyError(f"{label} missing required keys: {missing}")


def resolve_path(config_path: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def params_from_config(config: dict[str, Any]) -> Params:
    require_keys(config, ["params"], label="config")
    params_section = config["params"]
    if not isinstance(params_section, dict):
        raise ValueError("'params' must be a JSON object")
    allowed = Params.__dataclass_fields__.keys()
    kwargs = {key: params_section[key] for key in allowed if key in params_section}
    return Params(**kwargs)


def json_dumps_pretty(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


__all__ = [
    "json_dumps_pretty",
    "load_json_config",
    "params_from_config",
    "require_keys",
    "resolve_path",
]
