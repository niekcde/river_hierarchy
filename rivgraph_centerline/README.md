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

## Status

Initial framework only. No processing code has been added yet.
