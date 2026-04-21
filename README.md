# river-hierarchy

This repository is being organized as a single project with separate
subproject folders for related river-network workflows.

## Project structure

```text
river-hierarchy/
  SWORD_gauge_match/
  rivgraph_centerline/
```

## Subprojects

`SWORD_gauge_match/` contains the existing gauge, SWORD, discharge, and
kinematic screening pipeline. It includes RivRetrieve gauge metadata and
time-series fetching, SWORD reach/node matching, GRDC catalogue matching,
flood event detection, kinematic screening, QA exports, and QGIS-oriented
outputs.

`rivgraph_centerline/` will contain the RivGraph-based centerline and network
extraction workflow. RivGraph work will stay isolated in this subproject and
may use a separate Python or conda environment, a local fork, or an editable
install while compatibility with newer Python versions is evaluated.

The two subprojects are currently separate stages. They can each keep their
own README and implementation details for now, and will be merged into a
single pipeline once the RivGraph centerline/network extraction stage is ready.

## Data and outputs

Large generated outputs are local-only and ignored by Git. This includes
subproject output directories such as `SWORD_gauge_match/outputs/`,
`SWORD_gauge_match/configs/outputs/`, and future `rivgraph_centerline/outputs/`
products.

Input data and generated products should be documented in the relevant
subproject README rather than committed directly to this repository.

## Branch plan

- `SWORD_gauge_match`: existing matching, discharge, event, and kinematic
  screening pipeline.
- `rivgraph_centerline`: RivGraph-based centerline and network extraction
  work.
- `main`: eventual merged project once both stages are ready to be integrated.
