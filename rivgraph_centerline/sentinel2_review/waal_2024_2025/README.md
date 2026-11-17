# Waal: two-year single-channel review example

This is a **candidate for human review**, not an accepted training label. The selected Waal reach near Druten / Dodewaard has a wide, well-defined main channel, with engineered groynes and some off-channel water. The package covers **1 January 2024 through 31 December 2025**. The smaller yellow review window covers 2.5 × 1.4 km around the river, while the contextual rasters cover 6.4 × 5.8 km.

## Open and inspect

Open `review/Waal_review.qgz` in QGIS, or inspect `review/review_tile.png` first. The project opens on the pending review window, with the two-year RGB composite and a transparent candidate water mask. Toggle between:

- **All-components mask:** per-date spectral-rule water occurrence ≥75%.
- **Largest-component candidate:** the largest eight-connected water component of that same binary mask. No smoothing or hole filling is applied.
- **Excluded components:** everything removed by the largest-component selection, retained in its own raster. `excluded_component_occurrence.tif` preserves its continuous occurrence values.
- Continuous occurrence, annual occurrence, observation counts and dated RGB references.

Disconnected water is not declared erroneous. The component-filtered version is an optional candidate, not a training target approved in advance. Weak or missing connections can detach legitimate channels.

## What this composite means

The original Amazon/Tefé calibrated per-acquisition scoring model was not available during preparation. **This package does not reproduce that model and contains no calibrated mean-water-probability raster.** It provides a transparent Sentinel-2-only baseline and the aligned input subsets needed to apply the original classifier later.

For each acquisition, the fixed spectral baseline identifies candidate water where **MNDWI > 0 and NDVI < 0.3**. The NDVI cutoff is a pragmatic vegetation screen, not a locally fitted or validated model. Water occurrence is the number of usable dates passing that rule divided by the number of usable dates. The threshold of 75% is applied to that occurrence, not to a mean probability.

`mean_mndwi_NOT_probability.tif` is a continuous spectral index in [-1, 1]; it must not be interpreted as a water probability. A separate SCL water-occurrence raster uses the provider's scene classification (class 6) on the same valid support. Neither baseline establishes ground-truth accuracy.

`rgb_median_reflectance.tif` is the bandwise median of valid observations; `rgb_median_display.tif` is a fixed display stretch (reflectance / 0.25, clipped to [0,1], gamma exponent 0.65). The RGB composite is for interpretation. Water is classified separately on each acquisition before aggregation. Annual occurrence and dated RGB layers help distinguish temporal variation from classification errors. The complete-calendar-year composite is not a constant-discharge or summer-only product.

## Observation quality and provenance

- Public Earth Search **Sentinel-2 Collection-1 L2A**, MGRS tile 31UFT. One acquisition per UTC date; when duplicates exist, the lower scene-wide cloud percentage determines the selection. Full catalog metadata and the per-date inventory are saved outside the portable review folder.
- All catalogued dates in 2024–2025 are assessed locally. Dates with no usable pixels are recorded and contribute no observations.
- SCL classes 4, 5 and 6 are valid. No-data, defective pixels, cast/cloud shadows, unclassified pixels, clouds, cirrus, snow and ice are excluded. Shadow/cloud/snow classes are buffered by four 10 m grid cells with the default cross-shaped dilation. SCL is nearest-neighbour resampled from its 20 m source.
- Reflectances use the scale and offset in the Collection-1 asset metadata. The older `sentinel-2-l2a` collection was avoided because its offset metadata has documented inconsistencies. Negative corrected reflectances are clipped to zero for this baseline. Invalid band values and undefined MNDWI denominators are excluded from the shared support.
- A common **EPSG:32631, 10 m grid** is used. SWIR comes from a 20 m band and is bilinearly resampled; output grid spacing does not create 10 m SWIR detail.
- At least **five usable observations** are required for composite masks and indices. Float NoData is -9999; binary NoData is 255. Valid land is 0, distinct from missing observations. RGB validity is stored as a raster mask.
- `review/summary.json` reports actual acquisition counts, pixel observation counts, monthly/annual coverage and component areas. These counts describe usable observations, not independent samples or balanced seasonal sampling.

## Turn an accepted tile into training data

The GeoPackage `review_annotations.gpkg` contains:

1. `review_tiles`: the proposed review window, initially `status=pending`. Set `baseline` to the exact raster being judged. After inspecting the entire polygon, mark it `accepted`, and record reviewer/date. An accepted tile with zero edits is a valid **keep-as-is** training example.
2. `manual_edits`: polygons with `action=add/remove`, `reason`, `confidence`, and notes. The suggested reasons are bank, bar/island, channel connection, and river selection. These are suggestions, not mandatory physical classes.
3. `uncertain_areas`: regions that must not become hard training labels.

For a fully accepted tile, the training target is its selected starting mask with confident edits applied. Unedited pixels inside that reviewed area are confirmed unchanged. Pixels outside accepted review areas, inside uncertain areas, or without adequate observations remain **unknown**. Merely having no edit polygon is not enough outside a reviewed area. Keep original inputs separate from target labels and split later model evaluation by river.

The tile is deliberately left pending; no user acceptance or measured accuracy is claimed.

## Reproduce

Use the existing geospatial Python environment:

```text
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python fetch.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python build.py
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m pytest test_build.py -q
```

Scripts resolve paths relative to their own directory. `fetch.py` requires network access and resumes already completed per-date caches. `build.py` refuses an incomplete inventory and does not overwrite an existing annotation GeoPackage. Do not regenerate an accepted review package in place; copy/version it first.

Generate the styled project using the installed QGIS Python with `PYTHONPATH=/Applications/QGIS.app/Contents/Resources/python`, `PROJ_LIB=/Applications/QGIS.app/Contents/Resources/proj`, and `QT_QPA_PLATFORM=offscreen`, running `qgis_project.py`. The project is reread to verify all layers are valid. Raster paths are relative within the review package.

## Method sources

- [Earth Search catalog documentation](https://github.com/Element84/earth-search)
- [Copernicus Sentinel-2 processing and scene classification](https://sentiwiki.copernicus.eu/web/s2-processing)
- [Xu (2006): MNDWI](https://doi.org/10.1080/01431160600589179)

The exact combined MNDWI/NDVI rule used here is an explicit baseline choice, not a claim that the cited paper validates it for this reach.

## This build

317 acquisition dates were assessed; 166 had at least one usable pixel. Per-pixel observation counts range from 27 to 97, with a median of 74. All pixels meet the five-observation minimum. The unfiltered 75% occurrence mask contains 3.0393 km² of water; the largest component contains 2.2763 km², and the remaining 0.7630 km² is retained in the exclusions layer. These are mask areas, not estimates of classification accuracy.

Five automated logic tests passed. Independent checks verified all date records, the exact observation counts, occurrence fractions, shared raster grids, and reconstruction of the unfiltered mask from the largest component plus exclusions. The saved QGIS project was reopened with all 15 layers valid. The annotation window is still pending and the manual-edit layer is empty.
