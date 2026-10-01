# Final results figures

Start with `notebooks/final_results_figures.ipynb`, using the
`river-hierarchy-rivgraph` Python environment. Run from the repository or any
subdirectory. The notebook creates fourteen figures and saves PNG/PDF copies and
`figure_sources.json` under `figures/` (local, ignored by Git).

The configuration cell controls basemap location, centerline type, crop padding,
colors, and export settings. The global map uses Natural Earth 1:10 million land
and ocean polygons with red markers for the 17 final example bounding boxes.
The local basemap files currently live in the PhD paper folder specified by
`BASEMAP_DIR`, rather than this repository.

The five detail plots share an extent around example 5, unit 35. They read the
requested 2020Q4 source raster from `/Volumes/PhD/river_hierarchy/data`, the
existing prepared/cleaned masks, and independent-collapse state `S035_unit_35`.
The external PhD volume must be mounted. Existing prepared/cleaned masks derive
from 2020Q1, as recorded in the run's `prepare_summary.json`; the notebook keeps
this difference explicit. It does not regenerate or modify analysis products.

Section 3 adds cleaned-mask and centerline figures without arrows for example 5
unit 35, example 56 unit 12, and example 2 compound bubble 5. Each export has
a companion sources JSON with membership, crop bounds, and input paths.

Section 4 reproduces the width-measurement style from
`RAPID/notebooks/width_investigation.ipynb` (`dt_view`): original unit-35
branches and collapsed channel with/without island fill. The Appendix / Width
review adds example 22 link 530 and a wider local view of example 5 compound
bubble 33. Current Parquet samples and lineage-selected collapsed links are
used; historical exports may reflect earlier data products.
