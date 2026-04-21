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
  src/
  notebooks/
  outputs/
  README.md
```

`outputs/` is for generated products and should stay ignored by Git.

## Input notes

RivGraph expects binary masks. Some source masks may be georeferenced float
rasters with values between 0 and 1, so they need thresholding and preparation
before processing. Manual cleaning is expected before batch runs.

Example source data for early testing:

```text
/Users/6256481/Desktop/PhD_icloud/projecten/river_hierarchy/niek_review_package/water_masks_sarl/sarl_river_07.tif
```

## Environment notes

Use a separate Python or conda environment for RivGraph work. RivGraph may need
compatibility fixes because older versions were written and tested against
older Python and dependency versions. The first target is Python 3.11, using a
local fork or editable install if needed.

## Status

Initial skeleton only. Add setup notes, configuration examples, notebooks, and
implementation code as the RivGraph workflow is developed.
