"""Audited refresh of derived state hydraulic-context products.

This module deliberately does not regenerate masks, RivGraph products, widths,
or topology.  It is for correcting a hydraulic-context configuration after a
completed experiment has already been materialized.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from network_variants.hydraulic_context import (
    HYDRAULIC_CONTEXT_MANIFEST,
    SlopeConfig,
    generate_state_hydraulic_context,
)


def _text_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return Path(text).expanduser().resolve() if text else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _state_links_path(row: pd.Series) -> Path:
    analysis = _text_path(row.get("analysis_links_path"))
    if analysis is not None and analysis.exists():
        return analysis
    variant = _text_path(row.get("variant_output_dir"))
    candidate = variant / "widths" / "links_with_width_families.gpkg" if variant else None
    if candidate is not None and candidate.exists():
        return candidate
    raise FileNotFoundError(f"State {row.get('state_id')} has no usable analysis-width link product.")


def _state_nodes_path(row: pd.Series) -> Path:
    path = _text_path(row.get("directed_nodes_path"))
    if path is None or not path.exists():
        raise FileNotFoundError(f"State {row.get('state_id')} has no usable directed-node product.")
    return path


def _backup_context(context_dir: Path, destination: Path) -> list[str]:
    copied: list[str] = []
    for name in ("link_hydraulic_context.csv", "node_hydraulic_context.csv", HYDRAULIC_CONTEXT_MANIFEST):
        source = context_dir / name
        if source.exists():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / name)
            copied.append(name)
    return copied


def refresh_experiment_hydraulic_contexts(
    experiment_dir: str | Path,
    *,
    slope_config: SlopeConfig,
    execute: bool = False,
    backup_existing: bool = True,
    include_base_state: bool = True,
    state_ids: set[str] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    """Refresh every complete state in one experiment with recorded provenance.

    A dry run performs all input validation but writes nothing.  Execution
    preserves the prior three context artifacts per state in a compact,
    timestamped audit backup before replacing the derived products.
    """
    root = Path(experiment_dir).expanduser().resolve()
    registry_path = root / "state_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"State registry was not found: {registry_path}")
    registry = pd.read_csv(registry_path, low_memory=False)
    required = {"state_id", "state_dir", "parent_state_id", "status"}
    missing = sorted(required.difference(registry.columns))
    if missing:
        raise ValueError(f"State registry is missing required columns: {missing}")

    complete = registry.loc[registry["status"].astype(str).eq("complete")].copy()
    targets = complete.copy()
    if not include_base_state:
        base_mask = targets["state_id"].astype(str).eq("S000_base")
        if "state_role" in targets.columns:
            base_mask = base_mask | targets["state_role"].astype(str).eq("base")
        targets = targets.loc[~base_mask].copy()
    if state_ids is not None:
        requested = {str(value).strip() for value in state_ids if str(value).strip()}
        if not requested:
            raise ValueError("state_ids was supplied but contains no usable state IDs.")
        available = set(targets["state_id"].astype(str))
        unknown = sorted(requested.difference(available))
        if unknown:
            raise ValueError(f"Requested state IDs are not eligible refresh targets: {unknown[:10]}")
        targets = targets.loc[targets["state_id"].astype(str).isin(requested)].copy()
    if targets.empty:
        scope = "non-base complete" if not include_base_state else "complete"
        raise ValueError(f"Experiment has no {scope} states to refresh: {root}")
    targets["_depth"] = pd.to_numeric(targets.get("depth", 0), errors="coerce").fillna(0).astype(int)
    targets = targets.sort_values(["_depth", "state_id"], kind="mergesort")

    config_record = json.dumps(asdict(slope_config), sort_keys=True)
    plans: list[dict[str, Any]] = []
    for _, row in targets.iterrows():
        state_id = str(row["state_id"])
        state_dir = _text_path(row["state_dir"])
        if state_dir is None or not state_dir.exists():
            raise FileNotFoundError(f"State {state_id} has no usable state directory.")
        context_dir = _text_path(row.get("hydraulic_context_output_dir")) or (state_dir / "hydraulic_context")
        links_path = _state_links_path(row)
        nodes_path = _state_nodes_path(row)
        hierarchy_dir = _text_path(row.get("hierarchy_output_dir"))
        if hierarchy_dir is not None and not hierarchy_dir.exists():
            hierarchy_dir = None
        prior_manifest = _load_json(context_dir / HYDRAULIC_CONTEXT_MANIFEST)
        plans.append(
            {
                "state_id": state_id,
                "parent_state_id": "" if pd.isna(row.get("parent_state_id")) else str(row.get("parent_state_id")),
                "depth": int(row["_depth"]),
                "state_dir": str(state_dir),
                "directed_links_path": str(links_path),
                "directed_nodes_path": str(nodes_path),
                "hierarchy_output_dir": str(hierarchy_dir) if hierarchy_dir is not None else "",
                "hydraulic_context_dir": str(context_dir),
                "prior_slope_config": json.dumps(prior_manifest.get("slope_config", {}), sort_keys=True),
                "requested_slope_config": config_record,
                "status": "planned" if not execute else "pending",
                "backup_dir": "",
                "backup_files": "",
                "error": "",
            }
        )

    report = pd.DataFrame.from_records(plans)
    if max_workers < 1:
        raise ValueError("max_workers must be at least one.")
    if not execute:
        return report

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_root = root / "hydraulic_context_refresh_backups" / stamp
    refreshed_contexts: dict[str, Path] = {}
    if not include_base_state:
        # The promoted base context is an intentionally fixed parent input, not
        # a refresh target. Seed it so direct children receive its newly
        # promoted slopes/provenance during their normal inheritance step.
        base_rows = complete.loc[complete["state_id"].astype(str).eq("S000_base")]
        if "state_role" in complete.columns:
            role_rows = complete.loc[complete["state_role"].astype(str).eq("base")]
            base_rows = pd.concat([base_rows, role_rows], ignore_index=True).drop_duplicates("state_id")
        for _, base_row in base_rows.iterrows():
            base_state_dir = _text_path(base_row.get("state_dir"))
            if base_state_dir is None:
                continue
            base_context = _text_path(base_row.get("hydraulic_context_output_dir")) or (base_state_dir / "hydraulic_context")
            if not base_context.exists():
                raise FileNotFoundError(f"Promoted base context is missing: {base_context}")
            refreshed_contexts[str(base_row["state_id"])] = base_context
    def refresh_one(index: int, plan: pd.Series, parent_context: Path | None) -> tuple[int, str, Path | None, str, str, str]:
        """Refresh one independent state; parent contexts are fixed before a depth wave starts."""
        state_id = str(plan["state_id"])
        context_dir = Path(str(plan["hydraulic_context_dir"])).resolve()
        backup_dir_text = ""
        backup_files = ""
        try:
            if backup_existing:
                backup_dir = backup_root / state_id
                copied = _backup_context(context_dir, backup_dir)
                backup_dir_text = str(backup_dir) if copied else ""
                backup_files = ",".join(copied)
            kwargs: dict[str, Any] = {}
            if parent_context is not None:
                kwargs["parent_link_context_path"] = parent_context / "link_hydraulic_context.csv"
            outputs = generate_state_hydraulic_context(
                directed_links_path=plan["directed_links_path"],
                directed_nodes_path=plan["directed_nodes_path"],
                hierarchy_output_dir=plan["hierarchy_output_dir"] or None,
                output_dir=context_dir,
                state_id=state_id,
                config=slope_config,
                **kwargs,
            )
            return index, state_id, outputs.output_dir, "refreshed", backup_dir_text, backup_files
        except Exception as exc:  # retain a complete report when one state fails
            return index, state_id, None, f"failed:{exc}", backup_dir_text, backup_files

    for depth in sorted(report["depth"].unique()):
        wave = report.index[report["depth"].eq(depth)].tolist()
        runnable: list[tuple[int, pd.Series, Path | None]] = []
        for index in wave:
            plan = report.loc[index]
            parent_state_id = str(plan["parent_state_id"] or "").strip()
            parent_context = refreshed_contexts.get(parent_state_id)
            if parent_state_id and parent_context is None:
                report.at[index, "status"] = "failed"
                report.at[index, "error"] = f"Parent context was not refreshed first: {parent_state_id}"
                continue
            runnable.append((index, plan, parent_context))
        with ThreadPoolExecutor(max_workers=min(max_workers, len(runnable) or 1)) as executor:
            futures = [executor.submit(refresh_one, *item) for item in runnable]
            for future in as_completed(futures):
                index, state_id, output_dir, result, backup_dir, backup_files = future.result()
                report.at[index, "backup_dir"] = backup_dir
                report.at[index, "backup_files"] = backup_files
                if result == "refreshed":
                    refreshed_contexts[state_id] = output_dir  # available to the next depth wave
                    report.at[index, "status"] = "refreshed"
                else:
                    report.at[index, "status"] = "failed"
                    report.at[index, "error"] = result.removeprefix("failed:")

    report_path = root / f"hydraulic_context_refresh_{stamp}.csv"
    report.to_csv(report_path, index=False)
    summary_path = root / f"hydraulic_context_refresh_{stamp}.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "hydraulic-context-refresh-v1",
                "generated_at": datetime.now(UTC).isoformat(),
                "experiment_dir": str(root),
                "execute": True,
                "backup_existing": bool(backup_existing),
                "include_base_state": bool(include_base_state),
                "requested_state_ids": sorted(state_ids) if state_ids is not None else [],
                "max_workers": int(max_workers),
                "backup_root": str(backup_root) if backup_existing else "",
                "slope_config": asdict(slope_config),
                "states_total": int(len(report)),
                "states_refreshed": int(report["status"].eq("refreshed").sum()),
                "states_failed": int(report["status"].eq("failed").sum()),
                "report_csv": str(report_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
