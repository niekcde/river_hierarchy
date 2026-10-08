from __future__ import annotations

import ast
import csv
import errno
import gzip
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import netCDF4
except ImportError:  # pragma: no cover - optional dependency in some envs
    netCDF4 = None
from scipy.io import netcdf_file
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import factorized, splu

from .hydrograph import HydrographMetricConfig, write_hydrograph_outputs
from .parquet_compat import parquet_safe_metadata_frame


DEFAULT_QOUT_NAME = "Qout_rapid_framework.nc"
LEGACY_QOUT_BASENAME = "Qout_MS_b82_20150101_20240531_GLDASv21_ens_dtR10800"


@dataclass(frozen=True, slots=True)
class RapidResultStorageConfig:
    """Retention policy for completed prepared RAPID states.

    Compact mode preserves a gzip-compressed Qout, a Parquet reach/subreach
    mapping, hydrograph products, and a manifest. It can then remove the
    regenerable prep directory only after all result products are published.
    """

    compact_qout: bool = False
    compact_geometry: bool = False
    retain_prepared_inputs: bool = True
    write_mapping_parquet: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nonempty_file(path: Path) -> bool:
    """Whether a durable RAPID artifact is present and non-empty."""
    return path.is_file() and path.stat().st_size > 0


def _remove_disposable_prep(prep_dir: Path) -> None:
    """Remove regenerable preparation safely on filesystems with delete races.

    External APFS volumes can report a child as already removed while
    ``shutil.rmtree`` is walking the directory (notably alongside ``._``
    metadata files).  A completed run must not become a failed run merely
    because that final, disposable cleanup sees such a race.
    """
    if not prep_dir.exists():
        return

    def ignore_already_missing(_function: object, _path: str, excinfo: object) -> None:
        error = excinfo if isinstance(excinfo, BaseException) else excinfo[1]  # type: ignore[index]
        if isinstance(error, FileNotFoundError):
            return
        raise error

    shutil.rmtree(prep_dir, onexc=ignore_already_missing)
    if prep_dir.exists():
        raise OSError(f"Disposable RAPID prep directory was not fully removed: {prep_dir}")


def _recover_published_result(
    *,
    state_id: str,
    prep_dir: Path,
    run_dir: Path,
    storage: RapidResultStorageConfig,
) -> dict[str, object] | None:
    """Recover a complete result bundle after a cleanup-interrupted restart.

    Compact publication intentionally removes preparation inputs.  On a
    subsequent invocation, their absence is therefore expected when all final
    products already exist; rerouting would be wasteful and could overwrite a
    valid result.  Require every artifact promised by the selected storage
    policy before treating that result as complete.
    """
    # A present preparation directory means this invocation deliberately
    # rebuilt inputs (for example after an upstream hydraulic-context refresh).
    # Never recover the previous run in that case: it was produced from an
    # older prep bundle and must be replaced by routing the current one.
    if prep_dir.exists():
        return None

    qout_gz = run_dir / f"{DEFAULT_QOUT_NAME}.gz"
    qout_nc = run_dir / DEFAULT_QOUT_NAME
    qout = qout_gz if _nonempty_file(qout_gz) else qout_nc
    if not _nonempty_file(qout):
        return None
    if storage.compact_qout and qout != qout_gz:
        return None

    outlet = run_dir / "outlet_hydrograph.csv"
    metrics_csv = run_dir / "hydrograph_metrics.csv"
    metrics_json = run_dir / "hydrograph_metrics.json"
    if not all(_nonempty_file(path) for path in (outlet, metrics_csv, metrics_json)):
        return None

    record: dict[str, object] = {
        "state_id": state_id,
        "rapid_prep_dir": str(prep_dir),
        "rapid_run_dir": str(run_dir),
        "qout_nc": str(qout),
        "status": "ran",
        "hydrograph_status": "computed",
        "outlet_hydrograph_csv": str(outlet),
        "hydrograph_metrics_csv": str(metrics_csv),
        "hydrograph_metrics_json": str(metrics_json),
        "recovered_existing_published_result": True,
        "rapid_prep_retained": prep_dir.exists(),
        "compact_qout": qout == qout_gz,
    }
    if storage.write_mapping_parquet:
        mapping = run_dir / "rapid_link_mapping.parquet"
        if not _nonempty_file(mapping):
            return None
        record["rapid_link_mapping_parquet"] = str(mapping)
        record["rapid_link_mapping_sha256"] = _sha256(mapping)
    if not storage.retain_prepared_inputs:
        geometry = run_dir / "rapid_reach_attributes.gpkg"
        if not _nonempty_file(geometry):
            return None
        record["rapid_reach_attributes_gpkg"] = str(geometry)
        record["rapid_reach_attributes_sha256"] = _sha256(geometry)
    return record


def _publish_compact_result(
    *,
    prep_dir: Path,
    run_dir: Path,
    qout_path: Path,
    storage: RapidResultStorageConfig,
) -> dict[str, object]:
    result: dict[str, object] = {"qout_nc": str(qout_path), "compact_qout": False}
    if storage.write_mapping_parquet:
        source = prep_dir / "rapid_link_attributes.csv"
        if not source.exists():
            raise FileNotFoundError(f"RAPID link mapping was not found: {source}")
        mapping = parquet_safe_metadata_frame(pd.read_csv(source, low_memory=False))
        mapping_path = run_dir / "rapid_link_mapping.parquet"
        mapping.to_parquet(mapping_path, index=False, compression="zstd")
        result["rapid_link_mapping_parquet"] = str(mapping_path)
        result["rapid_link_mapping_sha256"] = _sha256(mapping_path)
    if not storage.retain_prepared_inputs:
        reach_geometry = prep_dir / "rapid_reach_attributes.gpkg"
        if not reach_geometry.exists():
            raise FileNotFoundError(
                "RAPID reach geometry was not found; it is required for downstream "
                f"reach-wave analysis before removing prep: {reach_geometry}"
            )
        retained_geometry = run_dir / reach_geometry.name
        # The normal experiment work root is on the same volume as its run
        # output.  Move rather than copy the large reach geometry in that case:
        # copying temporarily doubles its footprint precisely when the storage
        # policy is trying to reclaim space. ``replace`` also repairs a partial
        # destination left by an interrupted publication.
        try:
            os.replace(reach_geometry, retained_geometry)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # A caller may deliberately place disposable prep on another
            # volume. Preserve the existing cross-device behavior in that rare
            # case, while keeping the zero-extra-space same-volume path above.
            shutil.copy2(reach_geometry, retained_geometry)
        if storage.compact_geometry:
            if not storage.write_mapping_parquet:
                raise ValueError("Compact geometry requires the full Parquet attribute mapping.")
            import geopandas as gpd
            geometry = gpd.read_file(retained_geometry, columns=["reach_id"])
            temporary_geometry = retained_geometry.with_name("compact_geometry.tmp.gpkg")
            geometry.to_file(temporary_geometry, driver="GPKG")
            os.replace(temporary_geometry, retained_geometry)
            result["geometry_attributes_in_mapping"] = True
        result["rapid_reach_attributes_gpkg"] = str(retained_geometry)
        result["rapid_reach_attributes_sha256"] = _sha256(retained_geometry)
    if storage.compact_qout:
        compressed = qout_path.with_suffix(qout_path.suffix + ".gz")
        with qout_path.open("rb") as source, gzip.open(compressed, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        result["qout_nc"] = str(compressed)
        result["qout_uncompressed_bytes"] = int(qout_path.stat().st_size)
        result["qout_compressed_bytes"] = int(compressed.stat().st_size)
        result["compact_qout"] = True
        qout_path.unlink()
    return result


def _write_failure_bundle(
    *,
    prep_dir: Path,
    run_dir: Path,
    error: Exception,
    failure_dir: Path | None = None,
) -> Path:
    failure_dir = failure_dir or (prep_dir.parent / "failures")
    failure_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rapid-run-failure-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "prep_dir": str(prep_dir),
        "run_dir": str(run_dir),
        "error_type": type(error).__name__,
        "error": str(error),
        "prep_manifest_sha256": _sha256(prep_dir / "rapid_prep_manifest.json")
        if (prep_dir / "rapid_prep_manifest.json").exists() else None,
    }
    signature = hashlib.sha256(
        f"{payload['error_type']}::{payload['error']}".encode("utf-8")
    ).hexdigest()[:16]
    path = failure_dir / f"rapid_run_failure_{signature}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def con_vec(con_csv: str | Path) -> tuple[list[int], list[list[int]], list[list[float]], list[int]]:
    river_ids: list[int] = []
    downstream_ids: list[list[int]] = []
    downstream_ratios: list[list[float]] = []
    downstream_counts: list[int] = []
    with open(con_csv, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        first = next(reader, None)
        if first is None:
            return river_ids, downstream_ids, downstream_ratios, downstream_counts
        ncol = len(first)
        csvfile.seek(0)
        for row in reader:
            count = 0
            row_ids: list[int] = []
            row_ratios: list[float] = []
            for col in range(1, ncol):
                entry = ast.literal_eval(row[col])
                if entry[0] != 0:
                    row_ids.append(int(entry[0]))
                    row_ratios.append(float(entry[1]))
                    count += 1
            if row_ratios and not np.isclose(sum(row_ratios), 1.0, atol=1e-6):
                raise ValueError(f"Downstream ratios for reach {row[0]} do not sum to 1.")
            river_ids.append(int(row[0]))
            downstream_ids.append(row_ids)
            downstream_ratios.append(row_ratios)
            downstream_counts.append(count)
    return river_ids, downstream_ids, downstream_ratios, downstream_counts


def bas_vec(bas_csv: str | Path) -> np.ndarray:
    frame = pd.read_csv(bas_csv, header=None)
    return frame.iloc[:, 0].to_numpy(dtype=np.int64)


def hsh_tbl(
    river_ids_total: list[int],
    river_ids_basin: np.ndarray,
) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    hash_total = {river_id: idx for idx, river_id in enumerate(river_ids_total)}
    hash_basin = {int(river_id): idx for idx, river_id in enumerate(river_ids_basin)}
    basin_to_total = np.array([hash_total[int(river_id)] for river_id in river_ids_basin], dtype=np.int32)
    return hash_total, hash_basin, basin_to_total


def net_mat(
    downstream_ids: list[list[int]],
    downstream_counts: list[int],
    downstream_ratios: list[list[float]],
    hash_total: dict[int, int],
    river_ids_basin: np.ndarray,
    hash_basin: dict[int, int],
    *,
    dtype=np.float32,
) -> csc_matrix:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for basin_idx, river_id in enumerate(river_ids_basin):
        total_idx = hash_total[int(river_id)]
        for downstream_idx in range(downstream_counts[total_idx]):
            downstream_id = downstream_ids[total_idx][downstream_idx]
            if downstream_id != 0 and downstream_id in hash_basin:
                rows.append(hash_basin[downstream_id])
                cols.append(basin_idx)
                values.append(downstream_ratios[total_idx][downstream_idx])
    return csc_matrix(
        (values, (rows, cols)),
        shape=(len(river_ids_basin), len(river_ids_basin)),
        dtype=dtype,
    )


def k_x_vec(kpr_csv: str | Path, xpr_csv: str | Path, basin_to_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_total = pd.read_csv(kpr_csv, header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    x_total = pd.read_csv(xpr_csv, header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    return k_total[basin_to_total], x_total[basin_to_total]


def ccc_mat(kpr_basin: np.ndarray, xpr_basin: np.ndarray, dt_routing: float) -> tuple[csc_matrix, csc_matrix, csc_matrix]:
    denominator = dt_routing / 2.0 + kpr_basin * (1.0 - xpr_basin)
    c1 = (dt_routing / 2.0 - kpr_basin * xpr_basin) / denominator
    c2 = (dt_routing / 2.0 + kpr_basin * xpr_basin) / denominator
    c3 = (-dt_routing / 2.0 + kpr_basin * (1.0 - xpr_basin)) / denominator
    return (
        diags(c1, format="csc", dtype=np.float64),
        diags(c2, format="csc", dtype=np.float64),
        diags(c3, format="csc", dtype=np.float64),
    )


def rte_mat(net: csc_matrix, c1: csc_matrix, c2: csc_matrix, c3: csc_matrix) -> tuple[csc_matrix, csc_matrix, csc_matrix]:
    identity_matrix = identity(net.shape[0], format="csc", dtype=np.int32)
    linear = identity_matrix - c1 * net
    q_external = c1 + c2
    q_out = c3 + c2 * net
    return linear, q_external, q_out


def read_inflow_netcdf(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    try:
        with netcdf_file(path, "r", mmap=False) as ds:
            river_ids = np.array(ds.variables["rivid"].data, dtype=np.int64).copy()
            time = np.array(ds.variables["time"].data, dtype=np.int64).copy()
            m3_riv = np.array(ds.variables["m3_riv"].data, dtype=np.float64).copy()
    except TypeError as exc:
        if "valid NetCDF 3 file" not in str(exc) or netCDF4 is None:
            raise
        with netCDF4.Dataset(path, "r") as ds:
            river_ids = np.array(ds.variables["rivid"][:], dtype=np.int64).copy()
            time = np.array(ds.variables["time"][:], dtype=np.int64).copy()
            m3_riv = np.array(ds.variables["m3_riv"][:], dtype=np.float64).copy()
    if len(time) < 2:
        raise ValueError("RAPID inflow.nc requires at least two timesteps.")
    dt_forcing = int(time[1] - time[0])
    return river_ids, time, m3_riv, dt_forcing


def stp_cor(dt_forcing: int, dt_routing: float) -> int:
    if not np.isfinite(dt_routing) or dt_routing <= 0:
        raise ValueError("Routing timestep must be finite and positive.")
    quotient = dt_forcing / dt_routing
    if not np.isclose(quotient, round(quotient), rtol=0, atol=1e-9):
        raise ValueError(
            f"The quotient of forcing dt ({dt_forcing}) and routing dt ({dt_routing}) must be an integer."
        )
    return int(round(quotient))


def mus_rte(
    linear: csc_matrix,
    q_external: csc_matrix,
    q_out: csc_matrix,
    n_substeps: int,
    q_out_initial: np.ndarray,
    q_external_avg: np.ndarray,
    solver,
    diagnostics: dict[str, float] | None = None,
    *,
    previous_external: np.ndarray | None = None,
    previous_inflow_matrix: csc_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Route held forcing; optionally correct the first substep's inlet history.

    q_external remains C1+C2. For an explicit upstream boundary, subtract
    C2*(current-previous) on the first substep only. Subsequent substeps have
    identical current and previous held inflow. Omission preserves legacy use.
    """
    if (previous_external is None) != (previous_inflow_matrix is None):
        raise ValueError("Supply both previous external inflow and its coefficient matrix.")
    q_out_final = q_out_initial.copy()
    q_out_avg = np.zeros_like(q_out_initial)
    for substep in range(n_substeps):
        rhs = q_external.dot(q_external_avg) + q_out.dot(q_out_final)
        if substep == 0 and previous_external is not None:
            rhs += previous_inflow_matrix.dot(previous_external - q_external_avg)
        q_out_final = solver(rhs)
        if diagnostics is not None:
            if not np.isfinite(q_out_final).all():
                raise ValueError("Routing produced nonfinite substep discharge.")
            diagnostics["minimum_substep_flow_cms"] = min(diagnostics["minimum_substep_flow_cms"], float(q_out_final.min()))
            diagnostics["maximum_substep_flow_cms"] = max(diagnostics["maximum_substep_flow_cms"], float(q_out_final.max()))
        q_out_avg += q_out_final
    q_out_avg = q_out_avg / float(n_substeps)
    return q_out_avg, q_out_final


def write_qout_netcdf(
    output_path: str | Path,
    river_ids_basin: np.ndarray,
    time: np.ndarray,
    qout: np.ndarray,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with netcdf_file(path, "w") as ds:
        ds.createDimension("time", len(time))
        ds.createDimension("rivid", len(river_ids_basin))
        time_var = ds.createVariable("time", "i4", ("time",))
        rivid_var = ds.createVariable("rivid", "i4", ("rivid",))
        qout_var = ds.createVariable("Qout", "f8", ("time", "rivid"))
        time_var[:] = time.astype(np.int32)
        rivid_var[:] = river_ids_basin.astype(np.int32)
        qout_var[:, :] = qout.astype(np.float64)
        ds.history = b"Created by RAPID shared Python engine"
    return path


def run_prepared_state(
    prep_dir: str | Path,
    *,
    routing_timestep_seconds: float | None = None,
    output_path: str | Path | None = None,
    return_qout: bool = False,
) -> Path | tuple[Path, np.ndarray]:
    prep_path = Path(prep_dir).expanduser().resolve()
    manifest_path = prep_path / "rapid_prep_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"RAPID prep manifest was not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    con_csv = prep_path / "rat_srt.csv"
    inflow_nc = prep_path / "inflow.nc"
    kpr_csv = prep_path / "kfc.csv"
    xpr_csv = prep_path / "xfc.csv"
    bas_csv = prep_path / "riv.csv"

    for required in (con_csv, inflow_nc, kpr_csv, xpr_csv, bas_csv):
        if not required.exists():
            raise FileNotFoundError(f"Required RAPID input file was not found: {required}")
    if con_csv.stat().st_size == 0:
        raise ValueError(f"Connectivity file is empty for prepared state: {con_csv}")

    river_ids_total, downstream_ids, downstream_ratios, downstream_counts = con_vec(con_csv)
    river_ids_basin = bas_vec(bas_csv)
    hash_total, hash_basin, basin_to_total = hsh_tbl(river_ids_total, river_ids_basin)
    network_dtype = manifest.get("prep_config", {}).get("network_dtype", "float32")
    if network_dtype not in {"float32", "float64"}:
        raise ValueError("Unknown network precision.")
    net = net_mat(downstream_ids, downstream_counts, downstream_ratios, hash_total, river_ids_basin, hash_basin,
                  dtype=np.dtype(network_dtype))

    kpr_basin, xpr_basin = k_x_vec(kpr_csv, xpr_csv, basin_to_total)
    if routing_timestep_seconds is None:
        routing_timestep_seconds = float(manifest["routing"]["routing_dt_seconds"])
    policy = manifest.get("prep_config", {}).get("coefficient_policy", "nonnegative")
    dt = float(routing_timestep_seconds)
    if policy not in {"nonnegative", "signed_current_inflow", "signed_inflow"}:
        raise ValueError(f"Unknown coefficient policy: {policy}")
    if not np.isfinite(xpr_basin).all() or (xpr_basin >= .5).any():
        raise ValueError("Routing requires finite X less than 0.5.")
    if policy != "signed_inflow" and (xpr_basin < 0).any():
        raise ValueError("Negative X requires the signed_inflow coefficient policy.")
    if not np.isfinite(kpr_basin).all() or (kpr_basin <= 0).any():
        raise ValueError("Routing requires finite positive K.")
    if not np.isfinite(dt) or dt <= 0 or (dt > 2*kpr_basin*(1-xpr_basin) + 1e-10).any():
        raise ValueError("Routing timestep violates the retained upper bound.")
    if policy == "nonnegative" and (dt < 2*kpr_basin*xpr_basin - 1e-10).any():
        raise ValueError("Routing timestep violates the nonnegative-coefficient policy.")
    c1, c2, c3 = ccc_mat(kpr_basin, xpr_basin, float(routing_timestep_seconds))
    linear, q_external, q_out = rte_mat(net, c1, c2, c3)

    inflow_river_ids, time, m3_riv, dt_forcing = read_inflow_netcdf(inflow_nc)
    if not np.array_equal(np.array(river_ids_total, dtype=np.int64), inflow_river_ids):
        raise ValueError("The river IDs in rat_srt.csv and inflow.nc do not match.")
    n_substeps = stp_cor(dt_forcing, float(routing_timestep_seconds))
    boundary_mode = manifest.get("prep_config", {}).get("boundary_mode", "legacy_held_external")
    if boundary_mode not in {"legacy_held_external", "upstream_two_time"}:
        raise ValueError(f"Unknown boundary mode: {boundary_mode}")
    if boundary_mode == "upstream_two_time":
        is_root = np.asarray(net.getnnz(axis=1) == 0)
        if np.any(m3_riv[:, basin_to_total][:, ~is_root] != 0):
            raise ValueError("Two-time upstream mode requires external forcing only at inlet reaches.")

    q_out_initial = np.zeros(len(river_ids_basin), dtype=np.float64)
    initial_mode = manifest.get("prep_config", {}).get("initial_flow_mode", "zero")
    if initial_mode == "steady_first_inflow":
        steady_rhs = m3_riv[0][basin_to_total] / float(dt_forcing)
        q_out_initial = factorized(identity(len(river_ids_basin), format="csc") - net)(steady_rhs)
        if not np.isfinite(q_out_initial).all() or (q_out_initial < 0).any():
            raise ValueError("Initial steady flow is invalid.")
    elif initial_mode != "zero":
        raise ValueError(f"Unknown initial flow mode: {initial_mode}")
    previous_external = (steady_rhs.copy() if initial_mode == "steady_first_inflow"
                         else np.zeros(len(river_ids_basin)))

    qout_records = np.zeros((len(time), len(river_ids_basin)), dtype=np.float64)
    solver = factorized(linear)
    _ = splu(linear)

    substep_audit = {"minimum_substep_flow_cms": float(q_out_initial.min()),
                     "maximum_substep_flow_cms": float(q_out_initial.max())}
    on_initial_plateau = initial_mode == "steady_first_inflow"
    for timestep_index in range(len(time)):
        q_external_avg = m3_riv[timestep_index][basin_to_total] / float(dt_forcing)
        if on_initial_plateau and np.array_equal(q_external_avg, steady_rhs):
            # Exact declared equilibrium: long constant spin-up intervals add
            # no dynamics and need no repeated sparse solves.
            qout_records[timestep_index, :] = q_out_initial
            previous_external = q_external_avg.copy()
            continue
        on_initial_plateau = False
        q_out_avg, q_out_initial = mus_rte(
            linear,
            q_external,
            q_out,
            n_substeps,
            q_out_initial,
            q_external_avg,
            solver,
            diagnostics=substep_audit,
            previous_external=previous_external if boundary_mode == "upstream_two_time" else None,
            previous_inflow_matrix=c2 if boundary_mode == "upstream_two_time" else None,
        )
        previous_external = q_external_avg.copy()
        qout_records[timestep_index, :] = q_out_avg

    qout_path = Path(output_path).expanduser().resolve() if output_path is not None else prep_path / DEFAULT_QOUT_NAME
    if not np.isfinite(qout_records).all():
        raise ValueError("Routing produced nonfinite discharge.")
    write_qout_netcdf(qout_path, river_ids_basin, time, qout_records)
    qout_path.with_suffix(".audit.json").write_text(json.dumps({
        "coefficient_policy": policy,
        "initial_flow_mode": initial_mode,
        "boundary_mode": boundary_mode,
        "network_dtype": network_dtype,
        "routing_dt_seconds": dt,
        "minimum_reported_flow_cms": float(qout_records.min()),
        "negative_reported_flow_count": int((qout_records < 0).sum()),
        **substep_audit,
        "scope": "every reach at every native routing substep, plus forcing-interval means",
        "flow_clipped": False,
    }, indent=2))
    if return_qout:
        return qout_path, qout_records
    return qout_path


def run_prepared_experiment(
    experiment_dir: str | Path,
    *,
    only_prepared: bool = True,
    hydrograph_config: HydrographMetricConfig | None = None,
    storage_config: RapidResultStorageConfig | None = None,
) -> pd.DataFrame:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry_path = experiment_path / "rapid_prep_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"RAPID prep registry was not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    storage = storage_config or RapidResultStorageConfig()
    run_rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        status = getattr(row, "status", "")
        if only_prepared and status != "prepared":
            continue
        prep_dir = Path(getattr(row, "rapid_prep_dir")).expanduser().resolve()
        registered_run_dir = getattr(row, "rapid_run_dir", "")
        if registered_run_dir is None or pd.isna(registered_run_dir) or not str(registered_run_dir).strip():
            run_dir = prep_dir.parent / "run"
        else:
            run_dir = Path(str(registered_run_dir)).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        recovered = _recover_published_result(
            state_id=str(getattr(row, "state_id")),
            prep_dir=prep_dir,
            run_dir=run_dir,
            storage=storage,
        )
        if recovered is not None:
            if not storage.retain_prepared_inputs and prep_dir.exists():
                _remove_disposable_prep(prep_dir)
                recovered["rapid_prep_retained"] = False
            run_rows.append(recovered)
            continue
        qout_path = run_dir / DEFAULT_QOUT_NAME
        try:
            output = run_prepared_state(prep_dir, output_path=qout_path)
            run_row = {
                "state_id": getattr(row, "state_id"),
                "rapid_prep_dir": str(prep_dir),
                "rapid_run_dir": str(run_dir),
                "qout_nc": str(output),
                "status": "ran",
            }
            try:
                hydrograph_outputs = write_hydrograph_outputs(
                    prep_dir,
                    output,
                    run_dir,
                    config=hydrograph_config,
                )
                run_row.update(hydrograph_outputs)
            except Exception as hydrograph_exc:  # pragma: no cover
                run_row.update(
                    {
                        "hydrograph_status": "failed",
                        "hydrograph_error": str(hydrograph_exc),
                        "outlet_hydrograph_csv": "",
                        "hydrograph_metrics_csv": "",
                        "hydrograph_metrics_json": "",
                    }
                )
            published = _publish_compact_result(
                prep_dir=prep_dir, run_dir=run_dir, qout_path=output, storage=storage
            )
            run_row.update(published)
            if not storage.retain_prepared_inputs:
                _remove_disposable_prep(prep_dir)
                run_row["rapid_prep_retained"] = False
            else:
                run_row["rapid_prep_retained"] = True
            run_rows.append(run_row)
        except Exception as exc:  # pragma: no cover
            failure_bundle = _write_failure_bundle(
                prep_dir=prep_dir,
                run_dir=run_dir,
                error=exc,
                failure_dir=run_dir.parent / "failures",
            )
            run_rows.append(
                {
                    "state_id": getattr(row, "state_id"),
                    "rapid_prep_dir": str(prep_dir),
                    "rapid_run_dir": str(run_dir),
                    "qout_nc": "",
                    "status": "failed",
                    "error": str(exc),
                    "hydrograph_status": "",
                    "hydrograph_error": "",
                    "outlet_hydrograph_csv": "",
                    "hydrograph_metrics_csv": "",
                    "hydrograph_metrics_json": "",
                    "failure_bundle": str(failure_bundle),
                }
            )
    run_registry = pd.DataFrame(run_rows)
    run_registry_path = experiment_path / "rapid_run_registry.csv"
    run_registry.to_csv(run_registry_path, index=False)
    manifest_path = experiment_path / "rapid_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment_dir": str(experiment_path),
                "states_total": int(len(run_registry)),
                "states_ran": int(run_registry["status"].eq("ran").sum()) if not run_registry.empty else 0,
                "states_failed": int(run_registry["status"].eq("failed").sum()) if not run_registry.empty else 0,
                "hydrograph_metrics_computed": int(run_registry["hydrograph_status"].eq("computed").sum()) if "hydrograph_status" in run_registry.columns else 0,
                "hydrograph_metrics_failed": int(run_registry["hydrograph_status"].eq("failed").sum()) if "hydrograph_status" in run_registry.columns else 0,
                "hydrograph_metric_config": (
                    asdict(hydrograph_config)
                    if hydrograph_config is not None
                    else None
                ),
                "storage_config": asdict(storage),
            },
            indent=2,
        )
    )
    return run_registry


def run_rapid(
    directory: str | Path,
    ROUTING_TIMESTEP_SECONDS: int = 10800,
    runType: str = "random",
    seed: int = 1,
    output_path: str | Path | None = None,
    return_qout: bool = False,
) -> str | tuple[str, np.ndarray] | None:
    """Compatibility wrapper for legacy synthetic RAPID runs."""

    prep_path = Path(directory).expanduser().resolve()
    con_csv = prep_path / "rat_srt.csv"
    inflow_nc = prep_path / "inflow.nc"
    kpr_csv = prep_path / "kfc.csv"
    xpr_csv = prep_path / "xfc.csv"
    bas_csv = prep_path / "riv.csv"

    for required in (con_csv, inflow_nc, kpr_csv, xpr_csv, bas_csv):
        if not required.exists():
            raise FileNotFoundError(f"Required RAPID input file was not found: {required}")
    if con_csv.stat().st_size == 0:
        return None

    river_ids_total, downstream_ids, downstream_ratios, downstream_counts = con_vec(con_csv)
    river_ids_basin = bas_vec(bas_csv)
    hash_total, hash_basin, basin_to_total = hsh_tbl(river_ids_total, river_ids_basin)
    net = net_mat(downstream_ids, downstream_counts, downstream_ratios, hash_total, river_ids_basin, hash_basin)

    kpr_basin, xpr_basin = k_x_vec(kpr_csv, xpr_csv, basin_to_total)
    c1, c2, c3 = ccc_mat(kpr_basin, xpr_basin, int(ROUTING_TIMESTEP_SECONDS))
    linear, q_external, q_out = rte_mat(net, c1, c2, c3)

    inflow_river_ids, time, m3_riv, dt_forcing = read_inflow_netcdf(inflow_nc)
    if not np.array_equal(np.array(river_ids_total, dtype=np.int64), inflow_river_ids):
        raise ValueError("The river IDs in rat_srt.csv and inflow.nc do not match.")
    n_substeps = stp_cor(dt_forcing, int(ROUTING_TIMESTEP_SECONDS))

    q_out_initial = np.zeros(len(river_ids_basin), dtype=np.float64)
    qout_records = np.zeros((len(time), len(river_ids_basin)), dtype=np.float64)
    solver = factorized(linear)
    _ = splu(linear)

    for timestep_index in range(len(time)):
        q_external_avg = m3_riv[timestep_index][basin_to_total] / float(dt_forcing)
        q_out_avg, q_out_initial = mus_rte(
            linear,
            q_external,
            q_out,
            n_substeps,
            q_out_initial,
            q_external_avg,
            solver,
        )
        qout_records[timestep_index, :] = q_out_avg

    default_name = f"{LEGACY_QOUT_BASENAME}{runType}_{seed}.nc"
    qout_path = Path(output_path).expanduser().resolve() if output_path is not None else prep_path / default_name
    write_qout_netcdf(qout_path, river_ids_basin, time, qout_records)

    if return_qout:
        return str(qout_path), qout_records
    return str(qout_path)


__all__ = [
    "DEFAULT_QOUT_NAME",
    "LEGACY_QOUT_BASENAME",
    "RapidResultStorageConfig",
    "HydrographMetricConfig",
    "bas_vec",
    "ccc_mat",
    "con_vec",
    "hsh_tbl",
    "k_x_vec",
    "mus_rte",
    "net_mat",
    "read_inflow_netcdf",
    "rte_mat",
    "run_prepared_experiment",
    "run_prepared_state",
    "run_rapid",
    "stp_cor",
    "write_qout_netcdf",
]
