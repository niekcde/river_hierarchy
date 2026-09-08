"""Equivalent-resistance adapter for compact island-only observations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from network_variants.equivalent_resistance_experiment import (
    load_manifest_equivalent_resistance_config,
    run_equivalent_resistance_experiment,
)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_island_equivalent_resistance(
    analysis_manifest_path: str | Path,
    *,
    base_experiment_dir: str | Path,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compare only the genuine island changes against their anchored base units.

    Each real island uses its recorded anchor unit as the nominal base corridor.
    The twelve mask-equivalent observations remain source-state references and
    are not recomputed.
    """
    analysis_path = Path(analysis_manifest_path).expanduser().resolve()
    base_dir = Path(base_experiment_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve() if output_dir else analysis_path.parent / "island_equivalent_resistance"
    output_root.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(analysis_path, low_memory=False)
    base_registry = pd.read_csv(base_dir / "state_registry.csv", low_memory=False)
    base_rows = base_registry.loc[
        base_registry["state_role"].astype(str).eq("base")
        & base_registry["status"].fillna("complete").astype(str).eq("complete")
    ]
    if base_rows.empty:
        raise ValueError(f"No complete base state exists in {base_dir}.")

    real = observations.loc[
        observations["analysis_backend"].astype(str).isin(
            {"island_full_raster", "island_local_patch"}
        )
    ].copy()
    registry_rows = [base_rows.iloc[0].to_dict()]
    for row in real.itertuples(index=False):
        variant_dir = Path(_text(row.analysis_variant_dir)).expanduser().resolve()
        directed_dir = Path(_text(row.analysis_directed_links_path)).expanduser().resolve().parent
        registry_rows.append({
            "state_id": str(row.island_id),
            "parent_state_id": str(base_rows.iloc[0]["state_id"]),
            "depth": 1,
            "mode": "independent-units",
            "state_role": "derived",
            "selection_type": "island",
            "collapsed_selection_label": str(row.island_id),
            "selected_unit_ids": str(int(row.source_unit_id)),
            "state_dir": str(Path(_text(row.analysis_state_dir)).expanduser().resolve()),
            "variant_output_dir": str(variant_dir),
            "directed_links_path": str(Path(_text(row.analysis_directed_links_path)).expanduser().resolve()),
            "analysis_links_path": str(Path(_text(row.analysis_routing_width_links_path)).expanduser().resolve()),
            "directed_nodes_path": str(Path(_text(row.analysis_directed_nodes_path)).expanduser().resolve()),
            "link_lineage_path": str(directed_dir / "link_lineage.csv"),
            "node_match_path": str(directed_dir / "node_match.csv"),
            "status": "complete",
            "anchor_bifurcation": row.anchor_bifurcation,
            "anchor_confluence": row.anchor_confluence,
        })

    workspace = output_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry_rows).to_csv(workspace / "state_registry.csv", index=False)
    (workspace / "experiment_manifest.json").write_text(
        json.dumps({"mode": "independent-units", "base_experiment_dir": str(base_dir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    config, _ = load_manifest_equivalent_resistance_config(base_dir)
    reference_kb_path = base_dir / "reference_section" / "reference_kb_summary.json"
    if str(config.kb_mode) == "based_reference_section":
        config = replace(config, reference_kb_summary=_read_json(reference_kb_path))
    result = run_equivalent_resistance_experiment(
        workspace,
        output_dir=output_root / "computed",
        equivalent_resistance_config=config,
    )

    comparison_lookup = result.comparison.set_index("state_id") if not result.comparison.empty else pd.DataFrame()
    records: list[dict[str, object]] = []
    for row in observations.itertuples(index=False):
        island_id = str(row.island_id)
        if str(row.analysis_backend) == "independent_unit_alias":
            records.append({
                "island_id": island_id,
                "equivalent_resistance_status": "referenced_existing_independent_state",
                "source_unit_id": row.source_unit_id,
                "source_state_dir": _text(row.canonical_state_dir),
                "comparison_state_id": "",
            })
        else:
            comparison = comparison_lookup.loc[island_id] if island_id in comparison_lookup.index else pd.Series(dtype=object)
            matched_k = pd.to_numeric(comparison.get("K_eq_matched_base"), errors="coerce")
            variant_k = pd.to_numeric(comparison.get("K_variant_realized"), errors="coerce")
            k_change_fraction = (
                float((variant_k - matched_k) / matched_k)
                if pd.notna(matched_k) and pd.notna(variant_k) and float(matched_k) != 0.0
                else None
            )
            records.append({
                "island_id": island_id,
                "equivalent_resistance_status": "computed_island_state",
                "source_unit_id": row.source_unit_id,
                "source_state_dir": "",
                "comparison_state_id": island_id,
                "comparison_valid_for_matched": comparison.get("comparison_valid_for_matched"),
                "K_eq_nominal_base": comparison.get("K_eq_nominal_base"),
                "K_variant_realized": comparison.get("K_variant_realized"),
                "K_change_fraction_matched": k_change_fraction,
                "Lambda_matched": comparison.get("Lambda_matched"),
                "Gamma_matched": comparison.get("Gamma_matched"),
            })
    observation_registry = pd.DataFrame(records)
    observation_registry.to_csv(output_root / "island_equivalent_resistance_observation_registry.csv", index=False)
    (output_root / "island_equivalent_resistance_manifest.json").write_text(
        json.dumps({
            "schema_version": "island-equivalent-resistance-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_manifest": str(analysis_path),
            "base_experiment_dir": str(base_dir),
            "n_observations": int(len(observation_registry)),
            "n_computed_island_states": int(len(real)),
            "n_alias_references": int(len(observation_registry) - len(real)),
            "computed_output_dir": str(output_root / "computed"),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return observation_registry
