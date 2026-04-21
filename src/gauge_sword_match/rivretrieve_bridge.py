from __future__ import annotations

import subprocess
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .utils import get_logger

LOGGER = get_logger("rivretrieve_bridge")

SUPPORTED_COUNTRY_FUNCTIONS = {
    "AU": "australia",
    "BR": "brazil",
    "CA": "canada",
    "CL": "chile",
    "FR": "france",
    "GB": "uk",
    "IE": "ireland",
    "JP": "japan",
    "UK": "uk",
    "US": "usa",
    "ZA": "southAfrica",
}


class RivRetrieveBackend:
    def fetch_metadata(self, config: AppConfig) -> Path:
        raise NotImplementedError

    def fetch_timeseries(self, config: AppConfig, station_table: Path) -> Path:
        raise NotImplementedError


@dataclass(slots=True)
class RScriptRivRetrieveBackend(RivRetrieveBackend):
    executable: str
    scripts_dir: Path

    def fetch_metadata(self, config: AppConfig) -> Path:
        output_path = config.gauges.metadata_output
        args = [
            "--countries",
            ",".join(config.gauges.countries),
            "--output",
            str(output_path),
            "--function-map",
            _serialize_function_map(config.gauges.country_function_map),
        ]
        self._run_script("fetch_rivretrieve_metadata.R", args)
        return output_path

    def fetch_timeseries(self, config: AppConfig, station_table: Path) -> Path:
        output_path = config.timeseries.output
        args = [
            "--input",
            str(station_table),
            "--output",
            str(output_path),
            "--variable",
            config.timeseries.variable,
            "--function-map",
            _serialize_function_map(config.gauges.country_function_map),
            "--max-retries",
            config.timeseries.max_retries,
            "--retry-backoff-seconds",
            config.timeseries.retry_backoff_seconds,
            "--station-pause-seconds",
            config.timeseries.station_pause_seconds,
            "--country-pause-seconds",
            config.timeseries.country_pause_seconds,
        ]
        if config.timeseries.start_date:
            args.extend(["--start-date", config.timeseries.start_date])
        if config.timeseries.end_date:
            args.extend(["--end-date", config.timeseries.end_date])
        self._run_script("fetch_rivretrieve_timeseries.R", args)
        return output_path

    def _run_script(self, script_name: str, args: list[str]) -> None:
        script_path = self.scripts_dir / script_name
        command = [self.executable, str(script_path), *(_stringify_arg(arg) for arg in args)]
        LOGGER.info("Running %s", " ".join(command))
        output_lines: list[str] = []
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    if line:
                        LOGGER.info(line)
                        output_lines.append(line)
            return_code = process.wait()

        if return_code != 0:
            error_message = output_lines[-1] if output_lines else "R script failed without output."
            raise RuntimeError(f"Failed to run {script_name}: {error_message}")


def build_backend(config: AppConfig) -> RivRetrieveBackend:
    scripts_dir = Path(__file__).resolve().parents[2] / "r"
    return RScriptRivRetrieveBackend(executable=config.r.executable, scripts_dir=scripts_dir)


def _serialize_function_map(overrides: dict[str, str]) -> str:
    merged = {**SUPPORTED_COUNTRY_FUNCTIONS, **{key.upper(): value for key, value in overrides.items()}}
    return ",".join(f"{country}={function_name}" for country, function_name in sorted(merged.items()))


def _stringify_arg(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
