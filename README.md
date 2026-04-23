# river-hierarchy

This repository is being organized as a single project with separate
subproject folders for related river-network workflows.

## Project structure

```text
river-hierarchy/
  RAPID/
  SWORD_gauge_match/
  rivgraph_centerline/
  synthetic_runs/
```

## Subprojects

`SWORD_gauge_match/` contains the existing gauge, SWORD, discharge, and
kinematic screening pipeline. It includes RivRetrieve gauge metadata and
time-series fetching, SWORD reach/node matching, GRDC catalogue matching,
flood event detection, kinematic screening, QA exports, and QGIS-oriented
outputs.

`rivgraph_centerline/` will contain the RivGraph-based centerline and network
extraction workflow. RivGraph work will stay isolated in this subproject and
will use a separate conda environment. The RivGraph library itself should be
referenced from a fork of `VeinsOfTheEarth/RivGraph`, expected at
`niekcde/RivGraph`, so package-level fixes can be made without vendoring copied
source into this repository.

`synthetic_runs/` now serves as the synthetic-network subproject. The copied
legacy synthetic code is preserved there while the workflow is being extracted
into a cleaner `src/` layout with explicit provenance tracking.

`RAPID/` is the shared routing subproject. It will hold the reusable RAPID
engine and RAPID input-preparation code used by synthetic runs first and by
RivGraph-derived networks later.

These subprojects are currently separate stages. They can each keep their own
README and implementation details for now, and will be merged into a single
pipeline once the shared interfaces are stable.

## Data and outputs

Large generated outputs are local-only and ignored by Git. This includes
subproject output directories such as `SWORD_gauge_match/outputs/`,
`SWORD_gauge_match/configs/outputs/`, and future `rivgraph_centerline/outputs/`
products.

Input data and generated products should be documented in the relevant
subproject README rather than committed directly to this repository.

Local scratch copies of upstream packages, including
`rivgraph_centerline/RivGraph-master/`, are ignored. Use a pinned fork,
editable install, or deliberate submodule when the project needs to depend on
external source code.

## Branch plan

- `SWORD_gauge_match`: existing matching, discharge, event, and kinematic
  screening pipeline.
- `rivgraph_centerline`: RivGraph-based centerline and network extraction
  work.
- `synthetic_runs`: synthetic-network generation, routing experiments, and
  sensitivity workflows.
- `RAPID`: shared routing engine and RAPID input-preparation layer.
- `main`: eventual merged project once both stages are ready to be integrated.
