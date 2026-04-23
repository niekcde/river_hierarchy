# RivGraph centerline

Skeleton for the RivGraph-based centerline and river-network extraction stage
of the river hierarchy project.

## Goal

Build a workflow that starts from cleaned, georeferenced river masks and
extracts centerlines and network topology with RivGraph. This stage is separate
from `../SWORD_gauge_match/` for now and will be integrated with the SWORD
gauge matching, discharge, flood event, and kinematic screening pipeline later.

## Planned workflow

1. Prepare source river masks.
2. Threshold or otherwise convert masks to strict binary rasters.
3. Manually inspect and clean masks where needed.
4. Run RivGraph centerline and network extraction.
5. Export centerline, node, link, and QA products.
6. Define handoff products for the combined river hierarchy pipeline.

## Planned structure

```text
rivgraph_centerline/
  configs/
    rivgraph_centerline.example.yml
    sections.example.csv
  src/
    rivgraph_centerline/
  notebooks/
  outputs/
  environment.yml
  README.md
```

`outputs/` is for generated products and should stay ignored by Git.

## RivGraph dependency

RivGraph should be maintained as a separate library dependency, not copied into
this project as normal source files. The intended setup is:

```text
VeinsOfTheEarth/RivGraph  ->  niekcde/RivGraph
```

Use the `river-hierarchy` branch of `niekcde/RivGraph` for package-level fixes
if we encounter RivGraph bugs or need compatibility changes. Keep this
repository focused on the river hierarchy workflow: mask inventories, mask
preparation, QA, batch execution, and handoff products.

The local `RivGraph-master/` folder, if present, is treated as scratch/reference
source and is ignored by Git.

## Input notes

RivGraph expects binary masks. Some source masks may be georeferenced float
rasters with values between 0 and 1, so they need thresholding and preparation
before processing. Manual cleaning is expected before batch runs.

Example source data for early testing:

```text
/Users/6256481/Desktop/PhD_icloud/projecten/river_hierarchy/niek_review_package/water_masks_sarl/sarl_river_07.tif
```

## Manual edits

Manual cleaning is handled as vector corrections rather than direct raster
painting. Create a polygon GeoPackage with an `action` field:

- `add`: burn those pixels to water/channel value `1`
- `remove`: burn those pixels to background value `0`

Apply the edits to a prepared binary mask with:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python \
  rivgraph_centerline/src/rivgraph_centerline/manual_edits.py \
  --base-mask rivgraph_centerline/outputs/smoke_tests/sarl_river_07/masks_prepared/sarl_river_07_binary_projected.tif \
  --edits rivgraph_centerline/outputs/smoke_tests/sarl_river_07/manual_edits/sarl_river_07_manual_edits.gpkg \
  --output rivgraph_centerline/outputs/smoke_tests/sarl_river_07/masks_cleaned/sarl_river_07_cleaned.tif
```

The cleaned mask remains a strict uint8 binary GeoTIFF with the prepared mask's
CRS, transform, and dimensions.

## Recommended workflow

Use the staged runner:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/run_smoke_workflow.py --help
```

For a new mask named `my_test_01`, the normal steps are:

1. Prepare the binary/projected mask once:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/run_smoke_workflow.py prepare-mask \
  --name my_test_01 \
  --source-mask /path/to/my_mask.tif
```

This creates:

```text
rivgraph_centerline/outputs/smoke_tests/my_test_01/
  masks_prepared/my_test_01_binary_projected.tif
  manual_edits/my_test_01_manual_edits.gpkg   # expected QGIS edit path
  prepare_summary.json
```

2. Optionally create `manual_edits/my_test_01_manual_edits.gpkg` in QGIS and apply it:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/run_smoke_workflow.py apply-edits \
  --name my_test_01
```

3. Run RivGraph. This automatically prefers the cleaned mask if it exists; otherwise it uses the prepared mask:

```bash
MPLCONFIGDIR=/tmp/matplotlib-rivgraph \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/run_smoke_workflow.py run-rivgraph \
  --name my_test_01 \
  --exit-sides NS
```

For a single command workflow without manual editing:

```bash
MPLCONFIGDIR=/tmp/matplotlib-rivgraph \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/run_smoke_workflow.py run-all \
  --name my_test_01 \
  --source-mask /path/to/my_mask.tif \
  --exit-sides NS
```

`run-all` reuses the existing prepared mask unless you pass `--force-prepare`,
so you do not need to rebuild the binary mask after making manual edits.

## Environment notes

Use the separate conda environment defined in `environment.yml`. It targets
Python 3.12 and installs RivGraph from the expected fork:

```bash
conda env create -f environment.yml
conda activate river-hierarchy-rivgraph
```

Before creating the environment, fork `VeinsOfTheEarth/RivGraph` to
`niekcde/RivGraph` and create/push the `river-hierarchy` branch. Once the
workflow is reproducible, replace the floating branch reference in
`environment.yml` with a specific commit hash.

The environment installs the fork over HTTPS so setup does not require GitHub
SSH keys. Use SSH only if your local GitHub authentication is configured.

## A/B comparison

To compare your current fork branch against the smaller `jameshgrn/update`
patch set, create the second environment:

```bash
conda env create -f rivgraph_centerline/environment.update.yml
conda activate river-hierarchy-rivgraph-update
```

Run the same cleaned mask through both installs into separate output folders.
The current smoke-test script already accepts a custom mask and output folder:

```bash
MPLCONFIGDIR=/tmp/matplotlib-rivgraph \
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/smoke_test_sarl07.py \
  --mask rivgraph_centerline/outputs/smoke_tests/sarl_river_07/masks_cleaned/sarl_river_07_cleaned.tif \
  --output-dir rivgraph_centerline/outputs/smoke_tests/sarl_river_07_baseline \
  --exit-sides NS

MPLCONFIGDIR=/tmp/matplotlib-rivgraph \
/opt/anaconda3/envs/river-hierarchy-rivgraph-update/bin/python -B \
  rivgraph_centerline/smoke_tests/smoke_test_sarl07.py \
  --mask rivgraph_centerline/outputs/smoke_tests/sarl_river_07/masks_cleaned/sarl_river_07_cleaned.tif \
  --output-dir rivgraph_centerline/outputs/smoke_tests/sarl_river_07_update \
  --exit-sides NS
```

Then compare the two runs:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -B \
  rivgraph_centerline/smoke_tests/compare_smoke_runs.py \
  --baseline rivgraph_centerline/outputs/smoke_tests/sarl_river_07_baseline \
  --candidate rivgraph_centerline/outputs/smoke_tests/sarl_river_07_update \
  --output rivgraph_centerline/outputs/comparisons/sarl_river_07_ab_compare.json
```

The comparison report focuses on the parts most likely to move for river masks:
summary counts, link counts, node counts, and centerline/link lengths.

## Status

Framework plus first manual smoke-test tooling:

- binary-mask preparation and smoke-test runner
- vector-based manual edit application
- narrow A/B comparison helper for baseline vs update-branch runs
