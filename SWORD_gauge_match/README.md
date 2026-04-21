# gauge-sword-match

Python-first flood-study workflow for matching river gauge stations fetched through `RivRetrieve` to local SWORD river geometries.

The current implementation is deliberately conservative:

- Stage A searches against SWORD reach geometries.
- Stage B optionally refines the selected reach to the nearest SWORD node on that reach.
- Low-confidence or ambiguous matches are flagged instead of forced.

This is intended for flood-study crosswalk generation where false matches are more costly than leaving a gauge unmatched.

## Repository Layout

```text
project_root/
  pyproject.toml
  README.md
  configs/
    example.yml
  src/gauge_sword_match/
    cli.py
    config.py
    grdc_io.py
    rivretrieve_bridge.py
    sword_io.py
    gauge_io.py
    timeseries_io.py
    event_detection.py
    hydraulics.py
    kinematic_screen.py
    kinematic_qa.py
    spatial_index.py
    candidate_search.py
    scoring.py
    resolver.py
    qa_exports.py
    utils.py
  r/
    fetch_rivretrieve_metadata.R
    fetch_rivretrieve_timeseries.R
  tests/
```

## What The Pipeline Does

1. `fetch-gauges`
   - Runs `RivRetrieve` via `Rscript`.
   - Downloads station metadata for configured countries.
   - Normalizes the core fields:
     - `station_id`
     - `station_name`
     - `lat`
     - `lon`
     - `country`
     - `agency`
     - `drainage_area`
     - `river_name`

2. `build-crosswalk`
   - Cleans the gauge metadata.
   - Recursively scans the local SWORD parquet directory.
   - Ignores macOS AppleDouble files such as `._*.parquet`.
   - Loads only relevant reach subsets using:
     - recursive file discovery
     - file-level bbox metadata
     - lazy `pyarrow.dataset` scans
     - column selection
     - bbox filters
   - Builds per-region spatial indexes with `shapely.STRtree`.
   - Finds nearest reaches within a configurable radius.
   - Scores candidates with:
     - distance
     - river-name similarity
     - drainage-area consistency
     - ambiguity penalty
   - Resolves one best match per gauge.
   - Optionally refines matched reaches to the nearest SWORD node on that reach.
   - Exports QA tables and summary metrics.

3. `fetch-timeseries`
   - Runs `RivRetrieve` via `Rscript` for the configured station scope.
   - Groups requests by country/provider.
   - Retries transient provider failures with exponential backoff.
   - Falls back to a direct USGS daily-values request for US stations when `RivRetrieve::usa()` fails deterministically.
   - Writes merged timeseries output to CSV or Parquet.

4. `detect-events`
   - Standardizes RivRetrieve timeseries columns such as `Date` and `Q`.
   - Optionally merges in locally prepared GRDC daily timeseries when `grdc.timeseries_path` exists.
   - Applies the configured `timeseries.scope` to GRDC stations before reading the GRDC event input when `outputs/grdc/crosswalk_best.parquet` is available.
   - Processes station batches sequentially or in parallel with a live batch progress bar.
   - Writes intermediate batch outputs under `outputs/_event_batches/` so the full event run does not need to stay in memory.
   - Detects candidate flood events on each gauge hydrograph.
   - Computes:
     - `q0_pre_event_median`
     - `q0_event_start_discharge`
     - `t0_rise_t10_t90_hours`
     - `t0_rise_start_to_peak_hours`
     - event quality diagnostics
   - Keeps only high-quality events in `events_selected.parquet`.

5. `screen-kinematic`
   - Joins selected events to matched SWORD reaches.
   - Automatically combines the main crosswalk and the supplementary GRDC crosswalk when both are present.
   - Pulls configured reach attributes such as `width_obs_p50` and `slope_obs_p50`.
   - Processes screening batches sequentially or in parallel with a live batch progress bar.
   - Writes intermediate batch outputs under `outputs/_kinematic_batches/` and streams the final `kinematic_results.parquet` file to avoid holding the full result table in memory.
   - Repeats the hydraulic screening across:
     - multiple `kb` values
     - multiple `Q0` definitions
     - multiple `T0` definitions
   - Writes per-assumption results and per-station summaries.

## Supplementary GRDC Catalogue Matching

`match-grdc-catalog` is a separate auxiliary workflow for large-sample station discovery.

- Reads a downloaded GRDC station catalogue workbook.
- Filters to stations with daily discharge availability.
- Normalizes the GRDC fields to the same gauge schema used by the main matcher.
- Reuses the same SWORD candidate search, scoring, node refinement, and `high` / `medium` / `low` confidence classes.
- Writes outputs to a separate directory such as `outputs/grdc/` so the RivRetrieve products are not overwritten.
- Exports:
  - a GRDC crosswalk
  - a review queue
  - a QGIS package
  - a request-ready station table
  - a plain-text station-name list with duplicates preserved

This sits next to the main pipeline rather than inside the `fetch-timeseries` stage because the actual GRDC discharge request is still external and manual.

## SWORD Assumptions

The code is built around local SWORD parquet files such as:

- `/Volumes/PhD/SWORD/v17c/beta/parquet/*_reaches.parquet`
- `/Volumes/PhD/SWORD/v17c/beta/parquet/*_nodes.parquet`

Important notes:

- Reach candidate search is the first-stage matching target.
- Node matching is a refinement step after the best reach is selected.
- The loader supports both:
  - GeoParquet geometry metadata
  - WKB geometry columns such as `geometry` or `geom`
- For the supplied SWORD v17c beta files, the implementation already handles:
  - `geom` as WKB
  - region-partitioned files such as `af_*`, `na_*`, `eu_*`
  - reach bbox fields `x_min`, `x_max`, `y_min`, `y_max`

## Installation

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Core Python dependencies:

- `pandas`
- `pyarrow`
- `geopandas`
- `shapely`
- `pyproj`
- `PyYAML`
- `rapidfuzz`
- `openpyxl`
- `click`

### R

At minimum, install:

```r
install.packages("RivRetrieve")
```

Optional but recommended:

```r
install.packages("arrow")
```

Why `arrow` matters:

- The R metadata script can write Parquet only if `arrow` is installed.
- The R timeseries script can read Parquet station lists only if `arrow` is installed.
- If `arrow` is unavailable in R, use `.csv` paths in the YAML config for those inputs and outputs.

## Configuration

Start from [configs/example.yml](/Users/6256481/Documents/GitHub/river_hierarchy/configs/example.yml).

Key settings:

```yaml
sword:
  parquet_dir: /Volumes/PhD/SWORD/v17c/beta/parquet/
  search_radius_m: 5000

gauges:
  countries: ["US", "FR", "BR"]

grdc:
  catalog_path: /Volumes/PhD/river_hierarchy/data/GRDC_Stations 2.xlsx
  output_dir: ../outputs/grdc
  station_metadata_path: ../outputs/grdc_station_metadata.parquet
  timeseries_path: ../outputs/grdc_timeseries.parquet

matching:
  score_weights:
    distance: 0.6
    river_name: 0.2
    drainage_area: 0.2

kinematic:
  width_field: width_obs_p50
  slope_field: slope_obs_p50
  kb_values: [10, 20, 30, 40]
  event_runtime:
    batch_station_count: 250
    execution_mode: sequential
    workers: 4
  screen_runtime:
    batch_station_count: 100
    execution_mode: sequential
    workers: 4
```

Useful options:

- `sword.max_candidates`: maximum reach candidates kept per gauge before scoring
- `sword.use_node_refinement`: enable or disable nearest-node refinement
- `gauges.metadata_output`: raw RivRetrieve output path
- `gauges.metadata_path`: bypass `fetch-gauges` and use an existing file
- `gauges.country_function_map`: override ISO2-to-RivRetrieve function mapping
- `grdc.catalog_path`: path to the downloaded GRDC Excel catalogue
- `grdc.sheet_name`: workbook sheet to read, default `station_catalogue`
- `grdc.daily_only`: keep only GRDC stations with daily discharge availability
- `grdc.min_daily_years`: minimum GRDC daily-record length
- `grdc.output_dir`: separate output directory for GRDC matching artifacts
- `grdc.station_metadata_path`: local parquet written from GRDC download conversion, used to keep provider paths explicit
- `grdc.timeseries_path`: local GRDC daily discharge parquet that is merged into `detect-events` when present
- `timeseries.output`: output path for fetched timeseries
- `timeseries.scope`: station subset for timeseries fetch
  - `high_medium_matched_only` (default)
  - `matched_only`
  - `all_cleaned`
- `timeseries.max_retries`: extra retry attempts for transient provider errors
- `timeseries.retry_backoff_seconds`: base backoff used between retries
- `timeseries.station_pause_seconds`: pause between station requests
- `timeseries.country_pause_seconds`: pause between provider/country blocks
- `kinematic.width_field`: SWORD reach width attribute used in screening
- `kinematic.slope_field`: SWORD reach slope attribute used in screening
- `kinematic.kb_values`: width-depth relationship sensitivities
- `kinematic.q0_methods`: `Q0` reference discharge definitions
- `kinematic.t0_methods`: rising-limb `T0` definitions
- `kinematic.event_runtime.batch_station_count`: number of stations processed per `detect-events` batch
- `kinematic.event_runtime.execution_mode`: `sequential` or `parallel`
- `kinematic.event_runtime.workers`: worker count used when `detect-events` runs in parallel mode
- `kinematic.screen_runtime.batch_station_count`: number of stations processed per `screen-kinematic` batch
- `kinematic.screen_runtime.execution_mode`: `sequential` or `parallel`
- `kinematic.screen_runtime.workers`: worker count used when `screen-kinematic` runs in parallel mode

## CLI Usage

Fetch gauge metadata:

```bash
gauge-sword-match fetch-gauges --config configs/example.yml
```

Build the gauge-to-SWORD crosswalk:

```bash
gauge-sword-match build-crosswalk --config configs/example.yml
```

Build a separate GRDC-to-SWORD crosswalk from a downloaded GRDC catalogue:

```bash
gauge-sword-match match-grdc-catalog --config configs/example.yml
```

Export the matched gauges and matched SWORD reaches to a QGIS GeoPackage:

```bash
gauge-sword-match export-gpkg --config configs/example.yml
```

Fetch timeseries for the cleaned station list:

```bash
gauge-sword-match fetch-timeseries --config configs/example.yml
```

Detect high-quality flood events:

```bash
gauge-sword-match detect-events --config configs/example.yml
```

Optional runtime overrides for large event runs:

```bash
gauge-sword-match detect-events --config configs/example.yml --execution-mode parallel --workers 4 --batch-station-count 250
```

Screen matched reaches for provisional kinematic-wave compatibility:

```bash
gauge-sword-match screen-kinematic --config configs/example.yml
```

Optional runtime overrides for large screening runs:

```bash
gauge-sword-match screen-kinematic --config configs/example.yml --execution-mode parallel --workers 4 --batch-station-count 100
```

## Outputs

The pipeline writes to the configured output directory, defaulting to `outputs/`.

- `crosswalk_best.parquet`
  - one row per gauge
  - includes `reach_id`
  - includes `sword_node_id` when node refinement succeeds
  - includes score diagnostics and review flags

- `crosswalk_candidates.parquet`
  - candidate reach rows retained for each gauge
  - includes component scores and total score

- `gauges_cleaned.parquet`
  - cleaned station metadata used in matching

- `review_queue.parquet`
  - unmatched or low-confidence rows for manual inspection

- `summary_metrics.json`
  - gauge count
  - matched count
  - unmatched count
  - median match distance
  - low-confidence count

- `matched_qgis.gpkg`
  - `matched_gauges` point layer
  - `matched_reaches` line layer
  - ready to add directly in QGIS

- `events_all.parquet`
  - all detected candidate flood events
  - includes `Q0`, `T0`, prominence, rise-limb, and quality diagnostics

- `events_selected.parquet`
  - strict subset of high-quality events retained for hydraulic screening

- `event_summary.json`
  - event-detection counts and medians

- `_event_batches/`
  - one subdirectory per `detect-events` run
  - per-batch intermediate `events_all` and `events_selected` parquet files

- `_kinematic_batches/`
  - one subdirectory per `screen-kinematic` run
  - per-batch intermediate `kinematic_results` and `kinematic_summary` parquet files

- `kinematic_results.parquet`
  - one row per `station x event x kb x q0_method x t0_method`
  - includes `F0`, `Tplus`, and provisional kinematic classification

- `kinematic_summary.parquet`
  - one row per matched station summarizing stability across assumptions

- `kinematic_review_queue.parquet`
  - stations that are assumption-sensitive, multichannel, or hydraulically invalid

- `kinematic_metrics.json`
  - counts of screened stations, valid combinations, and provisional kinematic candidates

The supplementary GRDC workflow writes the same style of matching outputs into `outputs/grdc/` by default:

- `gauges_cleaned.parquet`
  - daily-record GRDC stations normalized to the matcher schema

- `crosswalk_candidates.parquet`
  - GRDC candidate reach rows retained for each station

- `crosswalk_best.parquet`
  - one resolved SWORD match per GRDC station, including confidence class

- `review_queue.parquet`
  - unmatched or low-confidence GRDC rows for manual inspection

- `matched_qgis.gpkg`
  - ready-to-review GRDC match layers for QGIS

- `request_stations.csv`
  - matched GRDC stations prepared for request submission, with station name, GRDC id, confidence class, and SWORD reach ids

- `request_station_names.txt`
  - newline-delimited GRDC station names with duplicates preserved

- `grdc_station_metadata.parquet`
  - locally converted GRDC station metadata keyed by `station_key`

- `grdc_timeseries.parquet`
  - locally converted GRDC daily discharge records merged into `detect-events` when present

## Matching Logic

### Stage A: Reach Candidate Search

For each gauge:

- compute a search bbox from the configured radius
- identify relevant SWORD reach files by file bbox
- lazily load only the needed reach subset
- query an in-memory spatial index
- retain nearest reaches within the radius

### Stage B: Optional Node Refinement

For each resolved reach match:

- load node rows for the same SWORD region and matched reach ids
- search only nodes belonging to the chosen reach
- record nearest `sword_node_id` when available

### Stage C: Scoring

Candidate scores combine:

- `distance`
- `river_name`
- `drainage_area`

Ambiguous candidate sets receive an additional penalty before final resolution.

### Stage D: Resolution

Per gauge, the best output includes:

- `reach_id`
- `sword_node_id`
- `distance_m`
- `total_score`
- `second_best_score`
- `score_gap`
- `confidence_class`
- `review_flag`

## QA And Manual Review

The workflow is intentionally biased toward avoiding false positives.

Current review behavior flags rows when:

- no reach is found within the search radius
- the selected match has low confidence
- the score gap to the runner-up is small
- the selected reach is still relatively far from the gauge
- node refinement did not produce a node id

This is appropriate for flood-study screening, where conservative matching is preferable to over-assigning gauges to the wrong river geometry.

## Tests

Run the synthetic unit tests with:

```bash
PYTHONPATH=src pytest
```

The current test suite covers:

- distance scoring
- fuzzy river-name scoring
- synthetic nearest-reach selection
- candidate resolution and unmatched handling

## Design Notes

- Python is the orchestrator.
- R is used only through `Rscript` subprocesses.
- The bridge is isolated in [rivretrieve_bridge.py](/Users/6256481/Documents/GitHub/river_hierarchy/src/gauge_sword_match/rivretrieve_bridge.py) so an `rpy2` backend can be added later without rewriting the matching code.
- Reach-based search is used first because it is simpler and safer for the initial version.
- Final outputs still include `sword_node_id` when node refinement succeeds.

## Known Limitations

- `RivRetrieve` field names vary by country backend, so the R metadata script uses heuristic normalization.
- Drainage-area consistency uses SWORD `facc` as a proxy. Depending on the region and upstream-area conventions, you may want to recalibrate this term.
- The current implementation is conservative and may leave some borderline stations unmatched.
- The default confidence thresholds are reasonable first-pass values, not tuned scientific parameters.
