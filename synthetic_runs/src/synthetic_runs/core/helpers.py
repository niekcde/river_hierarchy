"""Shared helper utilities extracted from the preserved synthetic legacy code."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Unserializable type: {type(obj)}")


def _edge_uid(u, v, key) -> str:
    return f"{u}->{v}::{key}"


def _grid_values(x0: float, x1: float, step: float) -> np.ndarray:
    return np.arange(x0, x1, step, dtype=float)


def _iter_width_splits_two(W_in: float, min_width: float, step: float):
    """Yield all valid two-way width splits under the discrete step rules."""
    W_in = float(W_in)
    min_width = float(min_width)
    step = float(step)

    start = math.ceil(min_width / step) * step
    end = math.floor((W_in - min_width) / step) * step
    if end < start:
        return

    n_steps = int(round((end - start) / step)) + 1
    for i in range(n_steps):
        w1 = start + i * step
        w2 = W_in - w1
        w1 = round(w1, 10)
        w2 = round(w2, 10)
        if w2 < min_width - 1e-9:
            continue
        if abs((w2 / step) - round(w2 / step)) > 1e-9:
            continue
        yield (w1, w2)


def _disjoint(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    a0, a1 = a
    b0, b1 = b
    return (a1 <= b0 + 1e-12) or (b1 <= a0 + 1e-12)


def _cross_loop_intersect(a: float, b: Tuple[float, float]) -> bool:
    b0, b1 = b
    return b0 <= a <= b1


def _crosses_intersect(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    xb1, xr1 = a
    xb2, xr2 = b
    return np.sign(xb1 - xb2) != np.sign(xr1 - xr2)


def _merge_summary_parts(parts_dir: Path, out_dir: Path, summary_format: str = "parquet") -> pd.DataFrame:
    """
    Merge chunk files from parts_dir into a single summary file in out_dir.
    """
    suffix = "parquet" if summary_format == "parquet" else "csv"
    parts = sorted(parts_dir.glob(f"summary_part_*.{suffix}"))
    if not parts:
        df = pd.DataFrame()
        if summary_format == "parquet":
            df.to_parquet(out_dir / "summary.parquet", index=False)
        else:
            df.to_csv(out_dir / "summary.csv", index=False)
        return df

    if summary_format == "parquet":
        dfs = [pd.read_parquet(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        df.to_parquet(out_dir / "summary.parquet", index=False)
    else:
        dfs = [pd.read_csv(p) for p in parts]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out_dir / "summary.csv", index=False)

    return df


__all__ = [
    "_cross_loop_intersect",
    "_crosses_intersect",
    "_disjoint",
    "_edge_uid",
    "_ensure_dir",
    "_grid_values",
    "_iter_width_splits_two",
    "_json_default",
    "_merge_summary_parts",
]
