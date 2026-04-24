"""Recipe and output IO helpers extracted from the preserved synthetic legacy code."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import List

import networkx as nx
import pandas as pd

from .helpers import _ensure_dir, _json_default
from .network import Params, RiverNetworkNX


def save_run_outputs(
    out_dir: str | Path,
    params: Params,
    summary_df: pd.DataFrame,
    networks: List[RiverNetworkNX],
    *,
    summary_format: str = "parquet",
    save_graph_pickles: bool = False,
    pickle_stride: int = 1,
):
    out = _ensure_dir(out_dir)

    meta_path = out / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(params.__dict__, f, indent=2, default=_json_default)

    if summary_format.lower() == "parquet":
        summary_path = out / "summary.parquet"
        summary_df.to_parquet(summary_path, index=False)
    elif summary_format.lower() == "csv":
        summary_path = out / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
    else:
        raise ValueError("summary_format must be 'parquet' or 'csv'")

    recipes_path = out / "networks.jsonl.gz"
    with gzip.open(recipes_path, "wt", encoding="utf-8") as gz:
        for net in networks:
            gz.write(json.dumps(net.to_recipe(), default=_json_default) + "\n")

    written = {
        "meta": str(meta_path),
        "summary": str(summary_path),
        "recipes": str(recipes_path),
    }

    if save_graph_pickles:
        gp_dir = _ensure_dir(out / "graphs_gpickle")
        for i, net in enumerate(networks):
            if i % max(1, int(pickle_stride)) != 0:
                continue
            nx.write_gpickle(net.G, gp_dir / f"G_{i:06d}.gpickle")
        written["gpickle_dir"] = str(gp_dir)

    return written


def load_network_by_id(recipes_gz_path: str | Path, network_id: int) -> RiverNetworkNX:
    with gzip.open(recipes_gz_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == network_id:
                recipe = json.loads(line)
                return RiverNetworkNX.from_recipe(recipe)
    raise ValueError(f"Network {network_id} not found in {recipes_gz_path}")


__all__ = ["load_network_by_id", "save_run_outputs"]
