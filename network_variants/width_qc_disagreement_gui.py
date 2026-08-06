from __future__ import annotations

import argparse
import json
import shutil
import sys
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox
from typing import Any, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from rasterio.features import shapes
from rasterio.windows import from_bounds
from shapely.geometry import box, shape

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from network_variants.width_qc_model import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_OUTPUT_DIR,
    FEATURE_COLUMNS,
    PREDICTIONS_FILENAME,
    SUMMARY_FILENAME,
)
from network_variants.width_junction_audit import (
    COMPARISONS_FILENAME as JUNCTION_COMPARISONS_FILENAME,
    CONSENSUS_FILENAME as JUNCTION_CONSENSUS_FILENAME,
    DEFAULT_OUTPUT_DIR as DEFAULT_JUNCTION_AUDIT_OUTPUT_DIR,
    MANIFEST_FILENAME as JUNCTION_AUDIT_MANIFEST_FILENAME,
)
from network_variants.width_review_archive import _atomic_write_csv, _atomic_write_json, _sha256
from network_variants.width_review_dataset import DEFAULT_REVIEW_ROOT
from network_variants.width_review_gui import (
    build_neighbor_lookup,
    compute_sample_qc_frame,
    local_neighbor_ids,
)


REVIEW_SCHEMA_VERSION = "width-qc-disagreement-review-v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "width_qc_disagreement_review"
REVIEW_FILENAME = "width_qc_disagreement_review_v1.csv"
MANIFEST_FILENAME = "width_qc_disagreement_review_manifest_v1.json"
DEFAULT_JUNCTION_WINDOW_ID = "p10_min250_max1000"
DEFAULT_JUNCTION_AUDIT_PATH = DEFAULT_JUNCTION_AUDIT_OUTPUT_DIR / JUNCTION_COMPARISONS_FILENAME
DEFAULT_JUNCTION_CONSENSUS_PATH = DEFAULT_JUNCTION_AUDIT_OUTPUT_DIR / JUNCTION_CONSENSUS_FILENAME
DEFAULT_MAX_JUNCTION_REVIEW_PRIORITY = 2

JUNCTION_EVIDENCE_COLUMNS = (
    "incoming_link_ids",
    "outgoing_link_ids",
    "incoming_max_link_length",
    "outgoing_max_link_length",
    "incoming_max_length_over_width",
    "outgoing_max_length_over_width",
    "incoming_endpoint_windows_nonoverlapping",
    "outgoing_endpoint_windows_nonoverlapping",
    "all_junction_endpoint_windows_nonoverlapping",
    "incoming_mean_width_sum",
    "outgoing_mean_width_sum",
    "incoming_junction_width_sum",
    "outgoing_junction_width_sum",
    "mean_continuity_ratio_recomputed",
    "mean_continuity_abs_deviation_recomputed",
    "junction_continuity_ratio",
    "junction_continuity_abs_deviation",
    "junction_minus_mean_abs_deviation",
    "absolute_deviation_change",
    "mean_high_junction_low",
    "junction_high_mean_low",
    "material_continuity_change",
)

ADJUDICATIONS = (
    "manual_upheld",
    "model_upheld",
    "ambiguous",
    "measurement_valid_but_routing_unsuitable",
    "data_or_geometry_problem",
    "exclude_from_training",
)
ADJUDICATION_LABELS = {
    "manual_upheld": "Manual upheld",
    "model_upheld": "Model upheld",
    "ambiguous": "Ambiguous",
    "measurement_valid_but_routing_unsuitable": "Measurement valid, routing unsuitable",
    "data_or_geometry_problem": "Data / geometry problem",
    "exclude_from_training": "Exclude from training",
}

CONTEXT_COLUMNS = (
    "upstream_continuity_signed_deviation",
    "downstream_continuity_ratio",
    "downstream_continuity_signed_deviation",
    "downstream_continuity_abs_deviation",
    "raw_wet_width",
    "raw_total_width",
    "raw_dry_width",
    "dry_fraction",
    "link_length",
    "width_sample_count",
    "n_invalid_width_samples",
    "n_trimmed_qc_samples",
    "lineage_type",
)


@dataclass(frozen=True)
class ReviewInputs:
    dataset_path: Path
    predictions_path: Path
    summary_path: Path
    dataset_sha256: str
    predictions_sha256: str
    summary_sha256: str
    model_name: str
    junction_audit_path: Path | None = None
    junction_audit_sha256: str | None = None
    junction_window_id: str | None = None
    junction_consensus_path: Path | None = None
    junction_consensus_sha256: str | None = None
    max_junction_review_priority: int = DEFAULT_MAX_JUNCTION_REVIEW_PRIORITY


@dataclass
class ExampleContext:
    example_id: str
    links: gpd.GeoDataFrame
    samples: pd.DataFrame
    mask_path: Path | None
    node_to_links: dict[int, set[int]]
    context_note: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normal_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def _as_bool(value: object) -> bool:
    return bool(value) if value is not None and not pd.isna(value) else False


def resolve_review_inputs(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    predictions_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    model_name: str | None = None,
    junction_audit_path: str | Path | None = DEFAULT_JUNCTION_AUDIT_PATH,
    junction_window_id: str | None = DEFAULT_JUNCTION_WINDOW_ID,
    junction_consensus_path: str | Path | None = DEFAULT_JUNCTION_CONSENSUS_PATH,
    max_junction_review_priority: int = DEFAULT_MAX_JUNCTION_REVIEW_PRIORITY,
) -> ReviewInputs:
    dataset = Path(dataset_path).expanduser().resolve()
    predictions = Path(
        predictions_path if predictions_path is not None else DEFAULT_MODEL_OUTPUT_DIR / PREDICTIONS_FILENAME
    ).expanduser().resolve()
    summary = Path(
        summary_path if summary_path is not None else DEFAULT_MODEL_OUTPUT_DIR / SUMMARY_FILENAME
    ).expanduser().resolve()
    for path in (dataset, predictions, summary):
        if not path.exists():
            raise FileNotFoundError(path)
    dataset_hash = _sha256(dataset)
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    resolved_model = model_name or summary_payload.get("provisional_candidate_model")
    if not resolved_model:
        raise ValueError("Could not determine the provisional model from the validation summary.")
    junction_audit: Path | None = None
    junction_hash: str | None = None
    junction_consensus: Path | None = None
    junction_consensus_hash: str | None = None
    if max_junction_review_priority < 1:
        raise ValueError("Maximum junction-review priority must be at least 1.")
    if junction_audit_path is not None:
        junction_audit = Path(junction_audit_path).expanduser().resolve()
        if not junction_audit.exists():
            raise FileNotFoundError(
                f"Near-junction audit not found at {junction_audit}. Run "
                "`python -m network_variants.width_junction_audit` first, or pass --no-junction-audit."
            )
        if not junction_window_id:
            raise ValueError("A junction window ID is required when a junction audit is supplied.")
        junction_hash = _sha256(junction_audit)
        junction_manifest_path = junction_audit.parent / JUNCTION_AUDIT_MANIFEST_FILENAME
        if junction_manifest_path.exists():
            junction_manifest = json.loads(junction_manifest_path.read_text(encoding="utf-8"))
            if junction_manifest.get("dataset_sha256") != dataset_hash:
                raise ValueError("Near-junction audit was generated from a different width-review dataset.")
            expected_audit_hash = (junction_manifest.get("file_sha256", {}) or {}).get("reviewed_comparisons")
            if expected_audit_hash and expected_audit_hash != junction_hash:
                raise ValueError("Near-junction comparison CSV does not match its audit manifest.")
        if junction_consensus_path is None:
            raise ValueError("A junction consensus path is required when junction evidence is enabled.")
        junction_consensus = Path(junction_consensus_path).expanduser().resolve()
        if not junction_consensus.exists():
            raise FileNotFoundError(junction_consensus)
        junction_consensus_hash = _sha256(junction_consensus)
        if junction_manifest_path.exists():
            expected_consensus_hash = (junction_manifest.get("file_sha256", {}) or {}).get("review_consensus")
            if expected_consensus_hash and expected_consensus_hash != junction_consensus_hash:
                raise ValueError("Junction consensus CSV does not match its audit manifest.")
    return ReviewInputs(
        dataset_path=dataset,
        predictions_path=predictions,
        summary_path=summary,
        dataset_sha256=dataset_hash,
        predictions_sha256=_sha256(predictions),
        summary_sha256=_sha256(summary),
        model_name=str(resolved_model),
        junction_audit_path=junction_audit,
        junction_audit_sha256=junction_hash,
        junction_window_id=junction_window_id if junction_audit is not None else None,
        junction_consensus_path=junction_consensus,
        junction_consensus_sha256=junction_consensus_hash,
        max_junction_review_priority=int(max_junction_review_priority),
    )


def load_junction_review_candidates(inputs: ReviewInputs) -> pd.DataFrame:
    columns = [
        "example_id",
        "id_link",
        "targeted_review_priority",
        "targeted_review_reason",
        "junction_result_stable_within_0_1",
    ]
    if inputs.junction_consensus_path is None:
        return pd.DataFrame(columns=columns)
    consensus = pd.read_csv(inputs.junction_consensus_path, dtype={"example_id": "string"})
    missing = sorted(set(columns).difference(consensus.columns))
    if missing:
        raise ValueError(f"Junction consensus is missing review-queue fields: {missing}")
    consensus["example_id"] = consensus["example_id"].str.zfill(2)
    consensus["id_link"] = pd.to_numeric(consensus["id_link"], errors="raise").astype(int)
    consensus["targeted_review_priority"] = pd.to_numeric(
        consensus["targeted_review_priority"], errors="raise"
    ).astype(int)
    selected = consensus.loc[
        consensus["targeted_review_priority"].le(inputs.max_junction_review_priority), columns
    ].copy()
    if selected.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Junction consensus contains duplicate example/link review candidates.")
    return selected


def attach_junction_evidence(frame: pd.DataFrame, inputs: ReviewInputs) -> pd.DataFrame:
    if inputs.junction_audit_path is None:
        return frame
    audit = pd.read_csv(inputs.junction_audit_path, dtype={"example_id": "string"})
    audit["example_id"] = audit["example_id"].str.zfill(2)
    required = {"example_id", "id_link", "window_id", *JUNCTION_EVIDENCE_COLUMNS}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"Near-junction audit is missing GUI evidence fields: {missing}")
    selected = audit.loc[audit["window_id"].eq(inputs.junction_window_id)].copy()
    if selected.empty:
        available = sorted(audit["window_id"].dropna().astype(str).unique())
        raise ValueError(
            f"Near-junction window {inputs.junction_window_id!r} was not found; available windows: {available}"
        )
    if selected.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Near-junction audit contains duplicate example/link rows for the selected window.")
    evidence_columns = ["example_id", "id_link", "window_id", *JUNCTION_EVIDENCE_COLUMNS]
    merged = frame.merge(
        selected[evidence_columns],
        on=["example_id", "id_link"],
        how="left",
        validate="1:1",
    )
    missing_evidence = merged["window_id"].isna()
    if missing_evidence.any():
        keys = merged.loc[missing_evidence, ["example_id", "id_link"]].to_dict("records")
        raise ValueError(f"Near-junction evidence is missing for disagreement rows: {keys}")
    return merged


def build_disagreement_frame(inputs: ReviewInputs) -> pd.DataFrame:
    dataset = pd.read_csv(inputs.dataset_path, dtype={"example_id": "string"})
    predictions = pd.read_csv(inputs.predictions_path, dtype={"example_id": "string"})
    dataset["example_id"] = dataset["example_id"].str.zfill(2)
    predictions["example_id"] = predictions["example_id"].str.zfill(2)
    model_predictions = predictions.loc[predictions["width_model_name"].eq(inputs.model_name)].copy()
    if model_predictions.empty:
        raise ValueError(f"No predictions found for model {inputs.model_name!r}.")
    junction_candidates = load_junction_review_candidates(inputs)
    model_disagreement = model_predictions["manual_model_disagreement"].map(_as_bool)
    if junction_candidates.empty:
        junction_key = pd.Series(False, index=model_predictions.index)
    else:
        candidate_index = pd.MultiIndex.from_frame(junction_candidates[["example_id", "id_link"]])
        prediction_index = pd.MultiIndex.from_frame(model_predictions[["example_id", "id_link"]])
        junction_key = pd.Series(prediction_index.isin(candidate_index), index=model_predictions.index)
    selected = model_predictions.loc[model_disagreement | junction_key].copy()
    if selected.duplicated(["example_id", "id_link"]).any():
        raise ValueError("Selected model predictions contain duplicate example/link keys.")
    selected = selected.merge(
        junction_candidates,
        on=["example_id", "id_link"],
        how="left",
        validate="1:1",
    )
    selected["junction_review_candidate"] = selected["targeted_review_priority"].notna()
    selected["review_trigger"] = np.select(
        [
            selected["manual_model_disagreement"].map(_as_bool) & selected["junction_review_candidate"],
            selected["manual_model_disagreement"].map(_as_bool),
            selected["junction_review_candidate"],
        ],
        ["model_disagreement_and_junction_audit", "model_disagreement", "junction_audit"],
        default="unknown",
    )

    # ``split_sensitive`` is already frozen into every held-out prediction
    # row. Do not merge the dataset copy as well: pandas would suffix the two
    # columns and leave the GUI without the canonical field name.
    dataset_feature_columns = [column for column in FEATURE_COLUMNS if column != "split_sensitive"]
    dataset_columns = [
        "example_id",
        "id_link",
        "variant_id",
        "id_us_node",
        "id_ds_node",
        *dataset_feature_columns,
        *CONTEXT_COLUMNS,
        "review_archive_dir",
        "review_targets_path",
        "review_manifest_path",
    ]
    missing = sorted(set(dataset_columns).difference(dataset.columns))
    if missing:
        raise ValueError(f"Training dataset is missing disagreement-review fields: {missing}")
    frame = selected.merge(
        dataset[dataset_columns],
        on=["example_id", "id_link"],
        how="left",
        validate="1:1",
    )
    if frame["variant_id"].isna().any():
        raise ValueError("One or more model disagreements could not be joined to the training dataset.")
    dataset_label = dataset.set_index(["example_id", "id_link"])["requires_correction"]
    expected = [dataset_label.loc[(row.example_id, row.id_link)] for row in frame.itertuples()]
    if not np.array_equal(np.asarray(expected, dtype=int), frame["requires_correction"].astype(int)):
        raise ValueError("Prediction labels disagree with the training dataset.")

    frame["disagreement_type"] = np.select(
        [
            frame["manual_model_disagreement"].map(_as_bool)
            & frame["requires_correction"].astype(int).eq(0),
            frame["manual_model_disagreement"].map(_as_bool)
            & frame["requires_correction"].astype(int).eq(1),
            frame["junction_review_candidate"].map(_as_bool),
        ],
        ["false_correction", "missed_correction", "junction_audit_candidate"],
        default="unknown",
    )
    frame["review_schema_version"] = REVIEW_SCHEMA_VERSION
    frame["adjudication"] = pd.NA
    frame["adjudicated_requires_correction"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["include_in_training"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame["review_comment"] = pd.NA
    frame["reviewer"] = pd.NA
    frame["reviewed_at"] = pd.NA
    frame = attach_junction_evidence(frame, inputs)
    return frame.sort_values(
        ["routing_impact_score", "example_id", "id_link"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def adjudication_result(
    adjudication: str | None,
    *,
    manual_requires_correction: int,
    model_requires_correction: int,
) -> tuple[int | None, bool | None]:
    if not adjudication or adjudication not in ADJUDICATIONS:
        return None, None
    if adjudication == "manual_upheld":
        return int(manual_requires_correction), True
    if adjudication == "model_upheld":
        return int(model_requires_correction), True
    if adjudication == "measurement_valid_but_routing_unsuitable":
        return 1, True
    if adjudication == "exclude_from_training":
        return None, False
    return None, None


def merge_existing_adjudications(frame: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return frame
    required = {"example_id", "id_link", "width_model_name", "adjudication"}
    missing = sorted(required.difference(existing.columns))
    if missing:
        raise ValueError(f"Existing disagreement review is missing fields: {missing}")
    old = existing.copy()
    old["example_id"] = old["example_id"].astype("string").str.zfill(2)
    old = old.loc[old["width_model_name"].eq(frame["width_model_name"].iloc[0])].copy()
    old = old.drop_duplicates(["example_id", "id_link"], keep="last")
    review_columns = [
        "example_id",
        "id_link",
        "adjudication",
        "adjudicated_requires_correction",
        "include_in_training",
        "review_comment",
        "reviewer",
        "reviewed_at",
    ]
    merged = frame.drop(columns=review_columns[2:]).merge(
        old[review_columns], on=["example_id", "id_link"], how="left", validate="1:1"
    )
    return merged


def _version_existing_outputs(output_dir: Path) -> None:
    existing = [output_dir / REVIEW_FILENAME, output_dir / MANIFEST_FILENAME]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return
    version_dir = output_dir / "versions" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.copy2(path, version_dir / path.name)


def write_disagreement_review(
    frame: pd.DataFrame,
    *,
    inputs: ReviewInputs,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    review_path = output / REVIEW_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    _version_existing_outputs(output)
    _atomic_write_csv(frame, review_path)
    reviewed = _normal_text(frame["adjudication"]).isin(ADJUDICATIONS)
    counts = frame.loc[reviewed, "adjudication"].value_counts().sort_index().to_dict()
    manifest = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "production_authorized": False,
        "model_name": inputs.model_name,
        "inputs": {
            "dataset": str(inputs.dataset_path),
            "dataset_sha256": inputs.dataset_sha256,
            "predictions": str(inputs.predictions_path),
            "predictions_sha256": inputs.predictions_sha256,
            "validation_summary": str(inputs.summary_path),
            "validation_summary_sha256": inputs.summary_sha256,
            "junction_audit": str(inputs.junction_audit_path) if inputs.junction_audit_path else None,
            "junction_audit_sha256": inputs.junction_audit_sha256,
            "junction_window_id": inputs.junction_window_id,
            "junction_consensus": str(inputs.junction_consensus_path) if inputs.junction_consensus_path else None,
            "junction_consensus_sha256": inputs.junction_consensus_sha256,
            "max_junction_review_priority": inputs.max_junction_review_priority,
        },
        "files": {"review": REVIEW_FILENAME},
        "file_sha256": {"review": _sha256(review_path)},
        "counts": {
            "disagreements": int(len(frame)),
            "review_cases": int(len(frame)),
            "model_disagreements": int(frame["manual_model_disagreement"].map(_as_bool).sum()),
            "junction_audit_candidates": int(frame["junction_review_candidate"].map(_as_bool).sum()),
            "junction_only_candidates": int(frame["review_trigger"].eq("junction_audit").sum()),
            "reviewed": int(reviewed.sum()),
            "unreviewed": int((~reviewed).sum()),
            "by_adjudication": {str(key): int(value) for key, value in counts.items()},
        },
        "label_contract": {
            "manual_upheld": "retain original manual requires_correction label",
            "model_upheld": "use held-out model prediction",
            "measurement_valid_but_routing_unsuitable": "requires correction for routing despite a valid local measurement",
            "ambiguous": "no training label until resolved",
            "data_or_geometry_problem": "no training label until source problem is resolved",
            "exclude_from_training": "explicitly omit from refitting",
        },
    }
    _atomic_write_json(manifest, manifest_path)
    return {"review": str(review_path), "manifest": str(manifest_path), **manifest["counts"]}


def _path_from_review_root(value: object, review_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (review_root / path).resolve()


def load_example_context(
    row: pd.Series,
    review_root: str | Path = DEFAULT_REVIEW_ROOT,
    *,
    disagreement_link_ids: Sequence[int] | None = None,
) -> ExampleContext:
    root = Path(review_root).expanduser().resolve()
    example_id = str(row["example_id"]).zfill(2)
    archive_dir = _path_from_review_root(row["review_archive_dir"], root)
    manifest_path = _path_from_review_root(row["review_manifest_path"], root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    files = manifest.get("files", {}) or {}
    links_path = archive_dir / str(files.get("links_snapshot", "reviewed_links_snapshot.gpkg"))
    links = gpd.read_file(links_path)
    for column in ("id_link", "id_us_node", "id_ds_node"):
        links[column] = pd.to_numeric(links[column], errors="coerce")
    links = links.dropna(subset=["id_link", "id_us_node", "id_ds_node"]).copy()
    links[["id_link", "id_us_node", "id_ds_node"]] = links[
        ["id_link", "id_us_node", "id_ds_node"]
    ].astype(int)

    notes: list[str] = []
    samples = pd.DataFrame()
    variant_manifest_path = Path(str(manifest.get("source", {}).get("variant_manifest", ""))).expanduser()
    if variant_manifest_path.exists():
        variant_manifest = json.loads(variant_manifest_path.read_text(encoding="utf-8"))
        output_paths = variant_manifest.get("output_paths", {}) or {}
        families_path = Path(str(output_paths.get("link_width_families", ""))).expanduser()
        samples_path = Path(str(output_paths.get("link_width_samples", ""))).expanduser()
        if families_path.exists():
            families = pd.read_csv(families_path)
            families["id_link"] = pd.to_numeric(families["id_link"], errors="coerce").astype("Int64")
            useful = [
                column for column in ("id_link", "wid_adj_wet", "wid_adj_total", "n_units")
                if column in families.columns
            ]
            if useful:
                overlap = [column for column in useful if column != "id_link" and column in links.columns]
                links = links.drop(columns=overlap, errors="ignore").merge(
                    families[useful].dropna(subset=["id_link"]), on="id_link", how="left", validate="1:1"
                )
        else:
            notes.append("width-family context unavailable")
        if samples_path.exists():
            raw_samples = pd.read_csv(samples_path)
            if disagreement_link_ids is not None and "id_link" in raw_samples.columns:
                wanted = {int(value) for value in disagreement_link_ids}
                sample_link_ids = pd.to_numeric(raw_samples["id_link"], errors="coerce")
                raw_samples = raw_samples.loc[sample_link_ids.isin(wanted)].copy()
            samples, _ = compute_sample_qc_frame(raw_samples)
        else:
            notes.append("two-bank sample overlay unavailable")
    else:
        notes.append("original variant manifest unavailable")

    mask_path: Path | None = None
    mask_record = manifest.get("fingerprints", {}).get("mask", {}) or {}
    mask_candidate = Path(str(mask_record.get("path", ""))).expanduser()
    if mask_candidate.exists():
        mask_path = mask_candidate.resolve()
    else:
        notes.append("review mask unavailable")
    return ExampleContext(
        example_id=example_id,
        links=links,
        samples=samples,
        mask_path=mask_path,
        node_to_links=build_neighbor_lookup(links),
        context_note="; ".join(notes),
    )


class WidthQcDisagreementGui:
    def __init__(
        self,
        root: tk.Tk,
        *,
        frame: pd.DataFrame,
        inputs: ReviewInputs,
        review_root: Path,
        output_dir: Path,
        reviewer: str | None,
        show_samples: bool,
    ) -> None:
        self.root = root
        self.frame = frame.copy()
        self.inputs = inputs
        self.review_root = review_root
        self.output_dir = output_dir
        self.default_reviewer = reviewer or ""
        self.show_samples = show_samples
        self.current_pos = self._first_unreviewed_position()
        self.context_cache: dict[str, ExampleContext] = {}
        self.saved = False
        self.map_view = "local"
        self.root.title(f"Width-QC disagreement review — {inputs.model_name}")
        self.root.protocol("WM_DELETE_WINDOW", self._close_without_save)
        self._build_ui()
        self._load_editor()
        self._draw()

    def _first_unreviewed_position(self) -> int:
        reviewed = _normal_text(self.frame["adjudication"]).isin(ADJUDICATIONS)
        positions = np.flatnonzero(~reviewed.to_numpy())
        return int(positions[0]) if len(positions) else 0

    def _build_ui(self) -> None:
        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        self.info_var = tk.StringVar()
        tk.Label(controls, textvariable=self.info_var, justify="left", anchor="w", wraplength=1450).pack(
            side=tk.TOP, fill=tk.X, padx=4, pady=(0, 5)
        )

        adjudication_row = tk.Frame(controls)
        adjudication_row.pack(side=tk.TOP, fill=tk.X)
        shortcuts = ["1", "2", "3", "4", "5", "6"]
        for shortcut, value in zip(shortcuts, ADJUDICATIONS):
            tk.Button(
                adjudication_row,
                text=f"{ADJUDICATION_LABELS[value]} [{shortcut}]",
                command=lambda selected=value: self._set_adjudication(selected, advance=True),
            ).pack(side=tk.LEFT, padx=2, pady=2)

        navigation_row = tk.Frame(controls)
        navigation_row.pack(side=tk.TOP, fill=tk.X, pady=(3, 2))
        tk.Button(navigation_row, text="Previous [←]", command=lambda: self._move(-1)).pack(side=tk.LEFT, padx=2)
        tk.Button(navigation_row, text="Next [→]", command=lambda: self._move(1)).pack(side=tk.LEFT, padx=2)
        tk.Button(navigation_row, text="Clear [0]", command=lambda: self._set_adjudication(None)).pack(side=tk.LEFT, padx=2)
        tk.Button(navigation_row, text="Zoom focal [Z]", command=lambda: self._set_zoom("focal")).pack(side=tk.LEFT, padx=8)
        tk.Button(navigation_row, text="Zoom local [X]", command=lambda: self._set_zoom("local")).pack(side=tk.LEFT, padx=2)
        tk.Button(navigation_row, text="Save [Ctrl+S]", command=self._save).pack(side=tk.LEFT, padx=8)
        tk.Button(navigation_row, text="Save and close [W]", command=self._save_and_close).pack(side=tk.LEFT, padx=2)

        editor_row = tk.Frame(controls)
        editor_row.pack(side=tk.TOP, fill=tk.X, pady=(3, 0))
        tk.Label(editor_row, text="Reviewer:").pack(side=tk.LEFT)
        self.reviewer_var = tk.StringVar(value=self.default_reviewer)
        tk.Entry(editor_row, textvariable=self.reviewer_var, width=18).pack(side=tk.LEFT, padx=(3, 12))
        tk.Label(editor_row, text="Comment:").pack(side=tk.LEFT)
        self.comment_var = tk.StringVar()
        tk.Entry(editor_row, textvariable=self.comment_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        self.figure = Figure(figsize=(16, 9), dpi=100)
        self.ax_map = self.figure.add_subplot(121)
        self.ax_metrics = self.figure.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Control-s>", lambda _event: self._save())

    def _on_key(self, event: Any) -> None:
        if event.widget.winfo_class() in {"Entry", "TEntry", "Text"}:
            return
        key = (event.keysym or "").lower()
        if key in {"1", "2", "3", "4", "5", "6"}:
            self._set_adjudication(ADJUDICATIONS[int(key) - 1], advance=True)
        elif key in {"0", "delete", "backspace"}:
            self._set_adjudication(None)
        elif key == "left":
            self._move(-1)
        elif key == "right":
            self._move(1)
        elif key == "z":
            self._set_zoom("focal")
        elif key == "x":
            self._set_zoom("local")
        elif key == "w":
            self._save_and_close()

    def _current_row(self) -> pd.Series:
        return self.frame.iloc[self.current_pos]

    def _store_editor(self) -> None:
        index = self.frame.index[self.current_pos]
        comment = self.comment_var.get().strip()
        reviewer = self.reviewer_var.get().strip()
        self.frame.at[index, "review_comment"] = comment if comment else pd.NA
        if pd.notna(self.frame.at[index, "adjudication"]):
            self.frame.at[index, "reviewer"] = reviewer if reviewer else pd.NA

    def _load_editor(self) -> None:
        row = self._current_row()
        self.comment_var.set("" if pd.isna(row.get("review_comment")) else str(row["review_comment"]))
        saved_reviewer = "" if pd.isna(row.get("reviewer")) else str(row["reviewer"])
        self.reviewer_var.set(saved_reviewer or self.reviewer_var.get() or self.default_reviewer)

    def _set_adjudication(self, adjudication: str | None, *, advance: bool = False) -> None:
        self._store_editor()
        index = self.frame.index[self.current_pos]
        row = self.frame.loc[index]
        label, include = adjudication_result(
            adjudication,
            manual_requires_correction=int(row["requires_correction"]),
            model_requires_correction=int(row["width_model_prediction"]),
        )
        self.frame.at[index, "adjudication"] = adjudication if adjudication else pd.NA
        self.frame.at[index, "adjudicated_requires_correction"] = label if label is not None else pd.NA
        self.frame.at[index, "include_in_training"] = include if include is not None else pd.NA
        self.frame.at[index, "reviewer"] = self.reviewer_var.get().strip() or pd.NA
        self.frame.at[index, "reviewed_at"] = _utc_now() if adjudication else pd.NA
        if advance and self.current_pos < len(self.frame) - 1:
            self.current_pos += 1
            self.map_view = "local"
            self._load_editor()
        self._draw()

    def _move(self, step: int) -> None:
        self._store_editor()
        self.current_pos = max(0, min(len(self.frame) - 1, self.current_pos + step))
        self.map_view = "local"
        self._load_editor()
        self._draw()

    def _set_zoom(self, mode: str) -> None:
        self.map_view = mode
        self._draw()

    def _context(self, row: pd.Series) -> ExampleContext:
        key = str(row["example_id"]).zfill(2)
        if key not in self.context_cache:
            disagreement_link_ids = self.frame.loc[
                self.frame["example_id"].astype("string").str.zfill(2).eq(key), "id_link"
            ].astype(int).tolist()
            self.context_cache[key] = load_example_context(
                row,
                self.review_root,
                disagreement_link_ids=disagreement_link_ids,
            )
        return self.context_cache[key]

    @staticmethod
    def _bounds(links: gpd.GeoDataFrame, focal: gpd.GeoDataFrame, samples: pd.DataFrame, mode: str) -> tuple[float, float, float, float]:
        source = focal if mode == "focal" and not focal.empty else links
        minx, miny, maxx, maxy = source.total_bounds
        if not samples.empty:
            xs = pd.concat([pd.to_numeric(samples.get(c, np.nan), errors="coerce") for c in ("x", "left_bank_x", "right_bank_x")])
            ys = pd.concat([pd.to_numeric(samples.get(c, np.nan), errors="coerce") for c in ("y", "left_bank_y", "right_bank_y")])
            xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
            if not xs.empty and not ys.empty:
                minx, miny = min(minx, xs.min()), min(miny, ys.min())
                maxx, maxy = max(maxx, xs.max()), max(maxy, ys.max())
        span = max(maxx - minx, maxy - miny, 1.0)
        pad = span * (0.22 if mode == "focal" else 0.15)
        return float(minx - pad), float(miny - pad), float(maxx + pad), float(maxy + pad)

    def _draw_mask(self, context: ExampleContext, bounds: tuple[float, float, float, float]) -> None:
        if context.mask_path is None:
            return
        bbox = box(*bounds)
        try:
            with rasterio.open(context.mask_path) as dataset:
                window = from_bounds(*bounds, transform=dataset.transform).round_offsets().round_lengths()
                data = dataset.read(1, window=window, boundless=True, fill_value=0)
                transform = dataset.window_transform(window)
            land = data == 0
            for mapping, value in shapes(land.astype(np.uint8), mask=land, transform=transform):
                if int(value) != 1:
                    continue
                geometry = shape(mapping).intersection(bbox)
                if not geometry.is_empty:
                    gpd.GeoSeries([geometry], crs=context.links.crs).plot(
                        ax=self.ax_map, facecolor="#e0e0e0", edgecolor="#969696", linewidth=0.6, zorder=0
                    )
        except (OSError, ValueError, rasterio.errors.RasterioError):
            return

    def _draw(self) -> None:
        self.ax_map.clear()
        self.ax_metrics.clear()
        row = self._current_row()
        context = self._context(row)
        link_id = int(row["id_link"])
        neighbor_ids = local_neighbor_ids(row, context.node_to_links)
        local = context.links.loc[context.links["id_link"].isin(neighbor_ids)].copy()
        focal = local.loc[local["id_link"].eq(link_id)]
        neighbors = local.loc[~local["id_link"].eq(link_id)]
        upstream_node = int(row["id_us_node"])
        incoming_neighbors = neighbors.loc[neighbors["id_ds_node"].eq(upstream_node)]
        outgoing_neighbors = neighbors.loc[neighbors["id_us_node"].eq(upstream_node)]
        junction_neighbor_ids = set(incoming_neighbors["id_link"]).union(outgoing_neighbors["id_link"])
        other_neighbors = neighbors.loc[~neighbors["id_link"].isin(junction_neighbor_ids)]
        samples = context.samples.loc[context.samples.get("id_link", pd.Series(dtype=int)).eq(link_id)].copy()
        if "used_in_trimmed_width" in samples:
            samples = samples.loc[samples["used_in_trimmed_width"].fillna(False)]
        bounds = self._bounds(local, focal, samples if self.show_samples else samples.iloc[0:0], self.map_view)
        self._draw_mask(context, bounds)
        if not other_neighbors.empty:
            other_neighbors.plot(ax=self.ax_map, color="#9ecae1", linewidth=2.0, zorder=2, label="other local links")
        if not incoming_neighbors.empty:
            incoming_neighbors.plot(ax=self.ax_map, color="#238b45", linewidth=3.0, zorder=3, label="incoming at upstream junction")
        if not outgoing_neighbors.empty:
            outgoing_neighbors.plot(ax=self.ax_map, color="#f16913", linewidth=3.0, zorder=3, label="other outgoing at upstream junction")
        if not focal.empty:
            focal.plot(ax=self.ax_map, color="#08306b", linewidth=4.5, zorder=4, label="focal link")
        for feature in local.itertuples(index=False):
            geometry = feature.geometry
            try:
                start = geometry.interpolate(0.43, normalized=True)
                end = geometry.interpolate(0.52, normalized=True)
                feature_id = int(feature.id_link)
                if feature_id == link_id:
                    color = "#252525"
                elif feature_id in set(incoming_neighbors["id_link"]):
                    color = "#006d2c"
                elif feature_id in set(outgoing_neighbors["id_link"]):
                    color = "#d94801"
                else:
                    color = "#3182bd"
                self.ax_map.annotate("", xy=(end.x, end.y), xytext=(start.x, start.y), arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5}, zorder=5)
                point = geometry.interpolate(0.7, normalized=True)
                width = getattr(feature, "wid_adj_wet", getattr(feature, "wid_adj", np.nan))
                width_text = f"{float(width):.1f}" if pd.notna(width) else "?"
                self.ax_map.text(point.x, point.y, f"{int(feature.id_link)}\n{width_text}", fontsize=8, ha="center", va="center", bbox={"facecolor": "white", "alpha": 0.8, "boxstyle": "round,pad=.15"}, zorder=6)
            except (AttributeError, ValueError, TypeError):
                continue
        if self.show_samples and not samples.empty:
            center_x = pd.to_numeric(samples.get("x"), errors="coerce")
            center_y = pd.to_numeric(samples.get("y"), errors="coerce")
            left_x = pd.to_numeric(samples.get("left_bank_x"), errors="coerce")
            left_y = pd.to_numeric(samples.get("left_bank_y"), errors="coerce")
            right_x = pd.to_numeric(samples.get("right_bank_x"), errors="coerce")
            right_y = pd.to_numeric(samples.get("right_bank_y"), errors="coerce")
            valid = (
                np.isfinite(center_x)
                & np.isfinite(center_y)
                & np.isfinite(left_x)
                & np.isfinite(left_y)
                & np.isfinite(right_x)
                & np.isfinite(right_y)
            )
            for cx, cy, lx, ly, rx, ry in zip(
                center_x[valid],
                center_y[valid],
                left_x[valid],
                left_y[valid],
                right_x[valid],
                right_y[valid],
            ):
                self.ax_map.plot(
                    [cx, lx],
                    [cy, ly],
                    color="#31a354",
                    alpha=0.55,
                    linewidth=1.0,
                    zorder=3,
                )
                self.ax_map.plot(
                    [cx, rx],
                    [cy, ry],
                    color="#de2d26",
                    alpha=0.55,
                    linewidth=1.0,
                    zorder=3,
                )
            self.ax_map.scatter(
                center_x[valid],
                center_y[valid],
                s=11,
                color="#252525",
                edgecolor="white",
                linewidth=0.25,
                zorder=7,
            )
            self.ax_map.scatter(left_x[valid], left_y[valid], s=18, color="#31a354", edgecolor="black", linewidth=.3, zorder=7)
            self.ax_map.scatter(right_x[valid], right_y[valid], s=18, color="#de2d26", edgecolor="black", linewidth=.3, zorder=7)
        self.ax_map.set_xlim(bounds[0], bounds[2])
        self.ax_map.set_ylim(bounds[1], bounds[3])
        self.ax_map.set_aspect("equal", adjustable="box")
        self.ax_map.set_title(f"Example {row['example_id']} — link {link_id} — {self.map_view} context")
        handles, labels = self.ax_map.get_legend_handles_labels()
        if handles:
            self.ax_map.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.85)

        adjudication = "unreviewed" if pd.isna(row["adjudication"]) else ADJUDICATION_LABELS.get(str(row["adjudication"]), str(row["adjudication"]))
        manual = "correct" if int(row["requires_correction"]) else "accept"
        model = "correct" if int(row["width_model_prediction"]) else "accept"
        reviewed_count = int(_normal_text(self.frame["adjudication"]).isin(ADJUDICATIONS).sum())
        self.info_var.set(
            f"Case {self.current_pos + 1}/{len(self.frame)} | reviewed {reviewed_count}/{len(self.frame)} | "
            f"example {row['example_id']} link {link_id} | trigger={row['review_trigger'].replace('_', ' ')} | "
            f"{row['disagreement_type'].replace('_', ' ')}\n"
            f"manual={manual} | held-out model={model} | P(correction)={float(row['width_model_probability']):.3f} "
            f"at threshold={float(row['width_model_threshold']):.2f} | adjudication={adjudication}"
            + (f" | {context.context_note}" if context.context_note else "")
        )
        if _as_bool(row.get("mean_high_junction_low", False)):
            junction_interpretation = "whole-link mismatch resolves near junction"
        elif _as_bool(row.get("junction_high_mean_low", False)):
            junction_interpretation = "new mismatch appears near junction"
        elif _as_bool(row.get("material_continuity_change", False)):
            junction_interpretation = "whole-link and junction evidence differ materially"
        else:
            junction_interpretation = "whole-link and junction evidence are broadly similar"
        incoming_local = "yes" if _as_bool(row.get("incoming_endpoint_windows_nonoverlapping", False)) else "no / overlapping"
        outgoing_local = "yes" if _as_bool(row.get("outgoing_endpoint_windows_nonoverlapping", False)) else "no / overlapping"
        junction_priority = row.get("targeted_review_priority", np.nan)
        junction_priority_text = "—" if pd.isna(junction_priority) else str(int(junction_priority))
        junction_reason = row.get("targeted_review_reason", "")
        junction_reason = "" if pd.isna(junction_reason) else str(junction_reason).replace("_", " ")
        lines = [
            "CLASSIFIER INPUTS",
            f"length / wet width       {float(row['length_over_raw_wet_width']):9.3f}",
            f"split sensitive          {bool(row['split_sensitive'])!s:>9}",
            f"opposition mean           {float(row['opposition_cos_trimmed_mean']):9.3f}",
            f"opposition median         {float(row['opposition_cos_trimmed_median']):9.3f}",
            f"invalid sample fraction   {float(row['invalid_sample_fraction']):9.3f}",
            f"upstream ratio            {float(row['upstream_continuity_ratio']):9.3f}",
            f"upstream abs deviation    {float(row['upstream_continuity_abs_deviation']):9.3f}",
            "",
            f"NEAR-JUNCTION AUDIT ({row.get('window_id', 'not loaded')})",
            f"incoming links            {str(row.get('incoming_link_ids', '')):>9}",
            f"outgoing links            {str(row.get('outgoing_link_ids', '')):>9}",
            f"whole mean widths       {float(row.get('incoming_mean_width_sum', np.nan)):7.1f} -> {float(row.get('outgoing_mean_width_sum', np.nan)):7.1f}",
            f"near-junction widths    {float(row.get('incoming_junction_width_sum', np.nan)):7.1f} -> {float(row.get('outgoing_junction_width_sum', np.nan)):7.1f}",
            f"whole-link abs deviation  {float(row.get('mean_continuity_abs_deviation_recomputed', np.nan)):9.3f}",
            f"junction abs deviation    {float(row.get('junction_continuity_abs_deviation', np.nan)):9.3f}",
            f"longest incoming length   {float(row.get('incoming_max_link_length', np.nan)):9.1f} m",
            f"incoming length / width   {float(row.get('incoming_max_length_over_width', np.nan)):9.3f}",
            f"incoming estimate local?  {incoming_local:>16}",
            f"outgoing estimates local? {outgoing_local:>16}",
            f"junction review priority  {junction_priority_text:>9}",
            f"audit trigger: {junction_reason or 'context only'}",
            f"interpretation: {junction_interpretation}",
            "",
            "INTERPRETATION CONTEXT",
            f"upstream signed deviation {float(row['upstream_continuity_signed_deviation']):9.3f}",
            f"downstream ratio          {float(row['downstream_continuity_ratio']):9.3f}",
            f"raw wet width             {float(row['raw_wet_width']):9.2f}",
            f"raw total width           {float(row['raw_total_width']):9.2f}",
            f"dry fraction              {float(row['dry_fraction']):9.3f}",
            f"link length               {float(row['link_length']):9.2f}",
            f"valid / total samples     {int(row['width_sample_count'] - row['n_invalid_width_samples']):4d} / {int(row['width_sample_count']):4d}",
            f"routing-impact proxy      {float(row['routing_impact_score']):9.2f}",
            "",
            "ADJUDICATION QUESTION",
            "Should this measured width be accepted or corrected for routing?",
            "Use 'measurement valid, routing unsuitable' when both statements are true.",
        ]
        self.ax_metrics.axis("off")
        self.ax_metrics.set_title("Evidence and decision semantics")
        self.ax_metrics.text(0, 1, "\n".join(lines), family="monospace", fontsize=9.7, va="top", linespacing=1.18)
        self.canvas.draw_idle()

    def _save(self) -> None:
        self._store_editor()
        result = write_disagreement_review(self.frame, inputs=self.inputs, output_dir=self.output_dir)
        self.saved = True
        messagebox.showinfo(
            "Disagreement review saved",
            f"Saved {result['reviewed']} of {result['disagreements']} adjudications.\n\n{result['review']}",
            parent=self.root,
        )

    def _save_and_close(self) -> None:
        self._store_editor()
        result = write_disagreement_review(self.frame, inputs=self.inputs, output_dir=self.output_dir)
        self.saved = True
        if result["unreviewed"] and not messagebox.askyesno(
            "Close partial review",
            f"{result['unreviewed']} disagreements remain unreviewed. The partial review is saved. Close anyway?",
            parent=self.root,
        ):
            return
        self.root.destroy()

    def _close_without_save(self) -> None:
        if messagebox.askyesno("Close without saving?", "Close the disagreement review without saving current changes?", parent=self.root):
            self.root.destroy()


def load_or_create_review(
    inputs: ReviewInputs,
    *,
    output_dir: str | Path,
    start_over: bool,
) -> pd.DataFrame:
    output = Path(output_dir).expanduser().resolve()
    frame = build_disagreement_frame(inputs)
    review_path = output / REVIEW_FILENAME
    manifest_path = output / MANIFEST_FILENAME
    if start_over or not review_path.exists():
        return frame
    if not manifest_path.exists():
        raise ValueError("Existing disagreement review has no manifest; use --start-over to preserve it and begin again.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("inputs", {}) or {}
    actual_hashes = {
        "dataset_sha256": inputs.dataset_sha256,
        "predictions_sha256": inputs.predictions_sha256,
        "validation_summary_sha256": inputs.summary_sha256,
    }
    mismatches = [name for name, value in actual_hashes.items() if expected.get(name) != value]
    # A review created before junction evidence existed can be upgraded in
    # place. Once its manifest records an audit, changing either the audit or
    # selected window invalidates automatic resume.
    if expected.get("junction_audit_sha256") is not None:
        if expected.get("junction_audit_sha256") != inputs.junction_audit_sha256:
            mismatches.append("junction_audit_sha256")
        if expected.get("junction_window_id") != inputs.junction_window_id:
            mismatches.append("junction_window_id")
        if expected.get("junction_consensus_sha256") is not None:
            if expected.get("junction_consensus_sha256") != inputs.junction_consensus_sha256:
                mismatches.append("junction_consensus_sha256")
            if expected.get("max_junction_review_priority") != inputs.max_junction_review_priority:
                mismatches.append("max_junction_review_priority")
    if mismatches:
        raise ValueError(
            "Existing adjudications were created from different inputs "
            f"({', '.join(mismatches)}); rerun with --start-over to preserve and replace them."
        )
    return merge_existing_adjudications(frame, pd.read_csv(review_path, dtype={"example_id": "string"}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adjudicate held-out manual/model width-QC disagreements without modifying archived reviews or widths."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--predictions", default=str(DEFAULT_MODEL_OUTPUT_DIR / PREDICTIONS_FILENAME))
    parser.add_argument("--validation-summary", default=str(DEFAULT_MODEL_OUTPUT_DIR / SUMMARY_FILENAME))
    parser.add_argument("--model-name", default=None, help="Defaults to the provisional candidate in the validation summary.")
    parser.add_argument(
        "--junction-audit",
        default=str(DEFAULT_JUNCTION_AUDIT_PATH),
        help="Reviewed-link near-junction comparison CSV created by width_junction_audit.",
    )
    parser.add_argument(
        "--junction-window-id",
        default=DEFAULT_JUNCTION_WINDOW_ID,
        help="Candidate endpoint window to display from the near-junction audit.",
    )
    parser.add_argument(
        "--junction-consensus",
        default=str(DEFAULT_JUNCTION_CONSENSUS_PATH),
        help="Cross-window targeted-review ranking created by width_junction_audit.",
    )
    parser.add_argument(
        "--max-junction-review-priority",
        type=int,
        default=DEFAULT_MAX_JUNCTION_REVIEW_PRIORITY,
        help="Include junction-audit candidates at or above this priority cutoff (default includes priorities 1 and 2).",
    )
    parser.add_argument(
        "--no-junction-audit",
        action="store_true",
        help="Launch without near-junction evidence (debugging/fallback only).",
    )
    parser.add_argument("--review-root", default=str(DEFAULT_REVIEW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--no-samples", action="store_true", help="Do not draw archived two-bank transect context.")
    parser.add_argument(
        "--start-over",
        action="store_true",
        help="Ignore saved adjudications. Existing output files are copied into versions/ on the next save.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = resolve_review_inputs(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        summary_path=args.validation_summary,
        model_name=args.model_name,
        junction_audit_path=None if args.no_junction_audit else args.junction_audit,
        junction_window_id=args.junction_window_id,
        junction_consensus_path=None if args.no_junction_audit else args.junction_consensus,
        max_junction_review_priority=args.max_junction_review_priority,
    )
    frame = load_or_create_review(inputs, output_dir=args.output_dir, start_over=args.start_over)
    if frame.empty:
        raise ValueError(f"Model {inputs.model_name!r} has no held-out disagreements to review.")
    root = tk.Tk()
    gui = WidthQcDisagreementGui(
        root,
        frame=frame,
        inputs=inputs,
        review_root=Path(args.review_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        reviewer=args.reviewer,
        show_samples=not args.no_samples,
    )
    root.mainloop()
    if gui.saved:
        print(f"Saved disagreement review to {Path(args.output_dir).expanduser().resolve() / REVIEW_FILENAME}")
    else:
        print("Review closed without saving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
