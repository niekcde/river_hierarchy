from __future__ import annotations

import ast
import csv
import json
from dataclasses import asdict
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


DEFAULT_QOUT_NAME = "Qout_rapid_framework.nc"
LEGACY_QOUT_BASENAME = "Qout_MS_b82_20150101_20240531_GLDASv21_ens_dtR10800"


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
        dtype=np.float32,
    )


def k_x_vec(kpr_csv: str | Path, xpr_csv: str | Path, basin_to_total: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k_total = pd.read_csv(kpr_csv, header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    x_total = pd.read_csv(xpr_csv, header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    return k_total[basin_to_total], x_total[basin_to_total]


def ccc_mat(kpr_basin: np.ndarray, xpr_basin: np.ndarray, dt_routing: int) -> tuple[csc_matrix, csc_matrix, csc_matrix]:
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


def stp_cor(dt_forcing: int, dt_routing: int) -> int:
    quotient = dt_forcing / dt_routing
    if round(quotient) != quotient:
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
) -> tuple[np.ndarray, np.ndarray]:
    q_out_final = q_out_initial.copy()
    q_out_avg = np.zeros_like(q_out_initial)
    for _ in range(n_substeps):
        rhs = q_external.dot(q_external_avg) + q_out.dot(q_out_final)
        q_out_final = solver(rhs)
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
    routing_timestep_seconds: int | None = None,
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
    net = net_mat(downstream_ids, downstream_counts, downstream_ratios, hash_total, river_ids_basin, hash_basin)

    kpr_basin, xpr_basin = k_x_vec(kpr_csv, xpr_csv, basin_to_total)
    if routing_timestep_seconds is None:
        routing_timestep_seconds = int(manifest["routing"]["routing_dt_seconds"])
    c1, c2, c3 = ccc_mat(kpr_basin, xpr_basin, int(routing_timestep_seconds))
    linear, q_external, q_out = rte_mat(net, c1, c2, c3)

    inflow_river_ids, time, m3_riv, dt_forcing = read_inflow_netcdf(inflow_nc)
    if not np.array_equal(np.array(river_ids_total, dtype=np.int64), inflow_river_ids):
        raise ValueError("The river IDs in rat_srt.csv and inflow.nc do not match.")
    n_substeps = stp_cor(dt_forcing, int(routing_timestep_seconds))

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

    qout_path = Path(output_path).expanduser().resolve() if output_path is not None else prep_path / DEFAULT_QOUT_NAME
    write_qout_netcdf(qout_path, river_ids_basin, time, qout_records)
    if return_qout:
        return qout_path, qout_records
    return qout_path


def run_prepared_experiment(
    experiment_dir: str | Path,
    *,
    only_prepared: bool = True,
    hydrograph_config: HydrographMetricConfig | None = None,
) -> pd.DataFrame:
    experiment_path = Path(experiment_dir).expanduser().resolve()
    registry_path = experiment_path / "rapid_prep_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"RAPID prep registry was not found: {registry_path}")
    registry = pd.read_csv(registry_path)
    run_rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        status = getattr(row, "status", "")
        if only_prepared and status != "prepared":
            continue
        prep_dir = Path(getattr(row, "rapid_prep_dir")).expanduser().resolve()
        run_dir = prep_dir.parent / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
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
            run_rows.append(run_row)
        except Exception as exc:  # pragma: no cover
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
