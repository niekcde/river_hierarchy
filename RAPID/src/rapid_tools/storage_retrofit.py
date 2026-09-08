"""Audited retrofit of existing RAPID results to the compact retention contract.

The retrofit is intentionally separate from :mod:`rapid_tools.engine`: it
does not rerun routing and it never touches failed or incomplete states. A
read-only inventory creates a manifest first. Its executor can later compact
only rows whose live files still satisfy the full analysis contract.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from .parquet_compat import parquet_safe_metadata_frame


RETROFIT_ACTION = "compact_qout_preserve_analysis_and_remove_prep"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocated_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except FileNotFoundError:
                continue
    return total


def _text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _path(value: object) -> Path | None:
    text = _text(value)
    return Path(text).expanduser() if text else None


def _result_paths_are_complete(row: pd.Series) -> tuple[bool, str]:
    if _text(row.get("status")) != "ran":
        return False, "run_status_not_ran"
    if _text(row.get("hydrograph_status")) != "computed":
        return False, "hydrograph_metrics_not_computed"
    for field in ("outlet_hydrograph_csv", "hydrograph_metrics_csv", "hydrograph_metrics_json"):
        path = _path(row.get(field))
        if path is None or not path.is_file():
            return False, f"missing_{field}"
    return True, "analysis_products_complete"


def _state_retrofit_row(experiment_dir: Path, row: pd.Series) -> dict[str, object]:
    state_id = _text(row.get("state_id"))
    prep_dir = _path(row.get("rapid_prep_dir"))
    run_dir = _path(row.get("rapid_run_dir"))
    qout_path = _path(row.get("qout_nc"))
    record: dict[str, object] = {
        "experiment": experiment_dir.name,
        "experiment_dir": str(experiment_dir),
        "state_id": state_id,
        "run_status": _text(row.get("status")),
        "hydrograph_status": _text(row.get("hydrograph_status")),
        "prep_dir": str(prep_dir) if prep_dir is not None else "",
        "run_dir": str(run_dir) if run_dir is not None else "",
        "qout_path": str(qout_path) if qout_path is not None else "",
        "outlet_hydrograph_csv": _text(row.get("outlet_hydrograph_csv")),
        "hydrograph_metrics_csv": _text(row.get("hydrograph_metrics_csv")),
        "hydrograph_metrics_json": _text(row.get("hydrograph_metrics_json")),
        "qout_bytes": int(qout_path.stat().st_size) if qout_path is not None and qout_path.is_file() else 0,
        "prep_bytes": _allocated_size_bytes(prep_dir) if prep_dir is not None and prep_dir.is_dir() else 0,
        "recommended_action": "",
        "eligibility": "",
        "rationale": "",
    }
    results_complete, result_reason = _result_paths_are_complete(row)
    if not results_complete:
        record.update({"recommended_action": "retain", "eligibility": "not_eligible", "rationale": result_reason})
        return record
    if prep_dir is None or run_dir is None or qout_path is None:
        record.update({"recommended_action": "retain", "eligibility": "not_eligible", "rationale": "missing_registered_path"})
        return record
    if not qout_path.is_file() or qout_path.suffix.lower() != ".nc":
        record.update({"recommended_action": "retain", "eligibility": "not_eligible", "rationale": "qout_not_existing_raw_netcdf"})
        return record
    if not prep_dir.is_dir():
        record.update({"recommended_action": "retain", "eligibility": "not_eligible", "rationale": "prep_not_available_for_audited_archive"})
        return record
    if prep_dir.name != "prep" or run_dir.name != "run" or prep_dir.parent != run_dir.parent:
        record.update({"recommended_action": "retain", "eligibility": "not_eligible", "rationale": "unexpected_rapid_directory_contract"})
        return record
    required_prep = ("rapid_reach_attributes.gpkg", "rapid_link_attributes.csv")
    missing = [name for name in required_prep if not (prep_dir / name).is_file()]
    if missing:
        record.update({"recommended_action": "retain", "eligibility": "not_eligible", "rationale": "missing_prep_analysis_artifact:" + ",".join(missing)})
        return record
    record.update(
        {
            "recommended_action": RETROFIT_ACTION,
            "eligibility": "eligible",
            "rationale": "complete_result_with_regenerable_prep_and_raw_qout",
        }
    )
    return record


@dataclass(frozen=True)
class RapidStorageRetrofit:
    rows: pd.DataFrame
    summary: dict[str, object]


def build_rapid_storage_retrofit_manifest(output_root: str | Path) -> RapidStorageRetrofit:
    """Read existing RAPID registries and classify states without changing them."""
    root = Path(output_root).expanduser().resolve()
    rows: list[dict[str, object]] = []
    for registry_path in sorted(root.glob("*/rapid_run_registry.csv")):
        experiment_dir = registry_path.parent
        try:
            registry = pd.read_csv(registry_path)
        except pd.errors.EmptyDataError:
            continue
        for _, row in registry.iterrows():
            rows.append(_state_retrofit_row(experiment_dir, row))
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["eligibility"].eq("eligible")] if not frame.empty else frame
    summary = {
        "schema_version": "rapid-storage-retrofit-manifest-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "n_rows": int(len(frame)),
        "n_eligible": int(len(eligible)),
        "raw_qout_bytes_eligible": int(eligible["qout_bytes"].sum()) if not eligible.empty else 0,
        "prep_bytes_eligible": int(eligible["prep_bytes"].sum()) if not eligible.empty else 0,
        "actions": {
            str(action): int(count)
            for action, count in frame["recommended_action"].value_counts().items()
        } if not frame.empty else {},
        "destructive_action_performed": False,
    }
    return RapidStorageRetrofit(rows=frame, summary=summary)


def write_rapid_storage_retrofit_manifest(output_root: str | Path) -> RapidStorageRetrofit:
    """Write the read-only retrofit inventory beside the output tree."""
    result = build_rapid_storage_retrofit_manifest(output_root)
    root = Path(output_root).expanduser().resolve()
    result.rows.to_csv(root / "rapid_storage_retrofit_manifest.csv", index=False)
    (root / "rapid_storage_retrofit_manifest.json").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _compact_one(row: pd.Series, *, execute: bool) -> dict[str, object]:
    prep_dir = Path(_text(row["prep_dir"]))
    run_dir = Path(_text(row["run_dir"]))
    qout_path = Path(_text(row["qout_path"]))
    result = {
        "qout_nc_gz": str(qout_path.with_suffix(qout_path.suffix + ".gz")),
        "rapid_reach_attributes_gpkg": str(run_dir / "rapid_reach_attributes.gpkg"),
        "rapid_link_mapping_parquet": str(run_dir / "rapid_link_mapping.parquet"),
        "bytes_reclaimed": 0,
    }
    if not execute:
        return result

    original_sha256 = _sha256(qout_path)
    qout_gz = Path(result["qout_nc_gz"])
    temporary_gz = qout_gz.with_name(f".{qout_gz.name}.{uuid.uuid4().hex}.tmp")
    try:
        with qout_path.open("rb") as source, gzip.open(temporary_gz, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        with gzip.open(temporary_gz, "rb") as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != original_sha256:
            raise ValueError("Gzip round-trip SHA-256 does not match raw Qout.")
        temporary_gz.replace(qout_gz)
    finally:
        if temporary_gz.exists():
            temporary_gz.unlink()

    reach_source = prep_dir / "rapid_reach_attributes.gpkg"
    mapping_source = prep_dir / "rapid_link_attributes.csv"
    shutil.copy2(reach_source, Path(result["rapid_reach_attributes_gpkg"]))
    mapping = parquet_safe_metadata_frame(pd.read_csv(mapping_source, low_memory=False))
    mapping.to_parquet(result["rapid_link_mapping_parquet"], index=False, compression="zstd")
    pd.testing.assert_frame_equal(mapping, pd.read_parquet(result["rapid_link_mapping_parquet"]))

    _update_run_registry(
        experiment_dir=Path(_text(row["experiment_dir"])),
        state_id=_text(row["state_id"]),
        original_qout=qout_path,
        compact_result=result,
    )
    raw_qout_bytes = int(qout_path.stat().st_size)
    prep_bytes = _allocated_size_bytes(prep_dir)
    retained_analysis_bytes = (
        int(qout_gz.stat().st_size)
        + int(Path(result["rapid_reach_attributes_gpkg"]).stat().st_size)
        + int(Path(result["rapid_link_mapping_parquet"]).stat().st_size)
    )
    reclaimed = raw_qout_bytes + prep_bytes - retained_analysis_bytes
    if reclaimed <= 0:
        raise ValueError(
            "Compact RAPID retention would not recover disk space for this state; "
            "raw Qout and prep were preserved."
        )
    qout_path.unlink()
    shutil.rmtree(prep_dir)
    result.update(
        {
            "qout_raw_sha256": original_sha256,
            "qout_gz_sha256": _sha256(qout_gz),
            "qout_raw_bytes": raw_qout_bytes,
            "qout_gz_bytes": int(qout_gz.stat().st_size),
            "prep_bytes_removed": prep_bytes,
            "retained_analysis_bytes": retained_analysis_bytes,
            "bytes_reclaimed": reclaimed,
        }
    )
    return result


def _update_run_registry(
    *,
    experiment_dir: Path,
    state_id: str,
    original_qout: Path,
    compact_result: dict[str, object],
) -> None:
    """Atomically point the state registry at its final compact artifacts."""
    registry_path = experiment_dir / "rapid_run_registry.csv"
    registry = pd.read_csv(registry_path)
    matches = registry["state_id"].astype(str).eq(str(state_id))
    if int(matches.sum()) != 1:
        raise ValueError(
            f"Expected exactly one rapid-run registry row for {state_id!r} in {registry_path}."
        )
    registered_qout = _path(registry.loc[matches, "qout_nc"].iloc[0])
    if registered_qout is None or registered_qout.resolve() != original_qout.resolve():
        raise ValueError("RAPID run registry Qout path changed since the retrofit manifest was written.")
    registry.loc[matches, "qout_nc"] = str(compact_result["qout_nc_gz"])
    registry.loc[matches, "rapid_reach_attributes_gpkg"] = str(
        compact_result["rapid_reach_attributes_gpkg"]
    )
    registry.loc[matches, "rapid_link_mapping_parquet"] = str(
        compact_result["rapid_link_mapping_parquet"]
    )
    registry.loc[matches, "rapid_prep_retained"] = False
    registry.loc[matches, "compact_qout"] = True
    temporary = registry_path.with_name(f".{registry_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        registry.to_csv(temporary, index=False)
        temporary.replace(registry_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_rapid_storage_retrofit(
    manifest_path: str | Path,
    *,
    execute: bool = False,
) -> RapidStorageRetrofit:
    """Dry-run or execute only live-validated eligible RAPID retrofit rows."""
    manifest = Path(manifest_path).expanduser().resolve()
    rows = pd.read_csv(manifest)
    reports: list[dict[str, object]] = []
    for index, row in rows.iterrows():
        base = {
            "manifest_path": str(manifest),
            "manifest_row": int(index),
            "experiment": _text(row.get("experiment")),
            "state_id": _text(row.get("state_id")),
            "execute_requested": bool(execute),
            "outcome": "",
            "detail": "",
            "bytes_reclaimed": 0,
        }
        if _text(row.get("recommended_action")) != RETROFIT_ACTION:
            reports.append({**base, "outcome": "out_of_scope_not_actioned", "detail": _text(row.get("rationale"))})
            continue
        # Rebuild the live classification from the persisted paths rather than
        # trusting a potentially stale manifest eligibility flag.
        live_row = row.rename(
            {
                "run_status": "status",
                "prep_dir": "rapid_prep_dir",
                "run_dir": "rapid_run_dir",
                "qout_path": "qout_nc",
            }
        )
        live = _state_retrofit_row(Path(_text(row["experiment_dir"])), live_row)
        if live["recommended_action"] != RETROFIT_ACTION:
            reports.append({**base, "outcome": "skipped", "detail": str(live["rationale"])})
            continue
        try:
            result = _compact_one(row, execute=execute)
            reports.append(
                {
                    **base,
                    **result,
                    "outcome": "compacted" if execute else "planned",
                    "detail": "Live analysis and round-trip integrity checks passed.",
                }
            )
        except Exception as exc:
            reports.append({**base, "outcome": "failed", "detail": f"{type(exc).__name__}: {exc}"})
    report_frame = pd.DataFrame(reports)
    execution_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_csv = manifest.with_name(manifest.stem + f"_execution_{execution_stamp}.csv")
    report_json = manifest.with_name(manifest.stem + f"_execution_{execution_stamp}.json")
    report_frame.to_csv(report_csv, index=False)
    summary = {
        "schema_version": "rapid-storage-retrofit-execution-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest),
        "execute_requested": bool(execute),
        "n_rows": int(len(report_frame)),
        "n_planned": int(report_frame["outcome"].eq("planned").sum()) if not report_frame.empty else 0,
        "n_compacted": int(report_frame["outcome"].eq("compacted").sum()) if not report_frame.empty else 0,
        "n_failed": int(report_frame["outcome"].eq("failed").sum()) if not report_frame.empty else 0,
        "bytes_reclaimed": int(report_frame["bytes_reclaimed"].sum()) if not report_frame.empty else 0,
        "execution_csv": str(report_csv),
        "destructive_action_performed": bool(execute and not report_frame.empty and report_frame["outcome"].eq("compacted").any()),
    }
    report_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return RapidStorageRetrofit(rows=report_frame, summary=summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit or safely retrofit existing RAPID results.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output-root")
    target.add_argument("--manifest-path")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root is not None:
        if args.execute:
            parser.error("--execute requires --manifest-path.")
        result = write_rapid_storage_retrofit_manifest(args.output_root)
    else:
        result = execute_rapid_storage_retrofit(args.manifest_path, execute=args.execute)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
