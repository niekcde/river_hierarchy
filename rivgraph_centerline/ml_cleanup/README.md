# Experimental river-mask cleaning

A reproducible supervised image-segmentation pilot, isolated from the production mask pipeline. Read [REPORT.md](REPORT.md) for findings and limitations. Models learn the existing cleaning policy; they do not establish whether every reference edit is physically correct.

## Reproduce

The existing geospatial environment is `/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python`. It supplies NumPy, SciPy, Rasterio, GeoPandas, scikit-image, scikit-learn, Matplotlib, Joblib and pytest. PyTorch 2.15.0 was installed separately in `/tmp/river-mask-ml-deps`; no package in the existing environment was changed. That temporary dependency directory is not durable. For future use, install these dependencies into a dedicated environment. The Torch wheel's bundled OpenMP library was replaced by a symlink to the existing environment's `libomp.dylib` in the temporary directory to prevent loading two OpenMP runtimes. The unsafe `KMP_DUPLICATE_LIB_OK` workaround was not used.

Run from the repository root, with these environment settings when using the temporary installation:

```bash
export PYTHONPATH=/tmp/river-mask-ml-deps
export MPLCONFIGDIR=/tmp/river-mask-mpl
export OPENBLAS_NUM_THREADS=1
```

Then run these commands with the Python interpreter above:

```text
python rivgraph_centerline/ml_cleanup/prepare.py
python rivgraph_centerline/ml_cleanup/audit.py
python -m pytest rivgraph_centerline/ml_cleanup/test_pipeline.py -q
python rivgraph_centerline/ml_cleanup/run.py
python rivgraph_centerline/ml_cleanup/stability.py
python rivgraph_centerline/ml_cleanup/verify_results.py
python rivgraph_centerline/ml_cleanup/report.py
python rivgraph_centerline/ml_cleanup/diagnostics.py
python rivgraph_centerline/ml_cleanup/whole_river.py
python rivgraph_centerline/ml_cleanup/finalize.py
```

`run.py` uses the Apple GPU when accessible and CPU otherwise. It resumes completed folds and existing tree models; to run a fresh experiment, use a separate copy/results directory, preserving the recorded experiment. Do not regenerate the dataset under a partially completed run. `--fold N` limits execution to one of four frozen folds. Seeds, river splits, validation thresholds, training history, cached-tile hashes and model artifacts are saved. Numerical GPU training is not promised to be bitwise deterministic.

## Apply to a new mask

`predict.py` accepts a saved U-Net, the prepared binary raster, original-valued source raster and its preparation threshold. It never reads a cleaned target. It writes a binary GeoTIFF and a probability-score GeoTIFF, preserves the prepared grid and refuses to overwrite existing predictions.

```text
python rivgraph_centerline/ml_cleanup/predict.py \
  --model rivgraph_centerline/ml_cleanup/results/fold_3/unet.pt \
  --prepared /path/to/prepared.tif \
  --source /path/to/original.tif \
  --source-threshold 20 \
  --threshold VALIDATION_SELECTED_THRESHOLD \
  --output /path/to/new_experimental_prediction.tif
```

Use the selected threshold from that fold's `selection.json`. The output scores are not calibrated confidence. All fold models are pilot models trained on only a subset of rivers; none has been installed into production. Match the source-value semantics and spatial resolutions to the audited training data.

## Scope and evaluation

- 15 eligible rivers, 840 native-resolution context tiles, four disjoint river-held-out folds.
- Input-selected river/water tiles are reported separately from manual-edit challenge patches.
- Baselines: unchanged, small-component/hole cleanup, binary spatial-feature Extra Trees, original-valued spatial-feature Extra Trees.
- CNN: small residual U-Net, boundary/edit-weighted binary cross-entropy, Dice and soft-skeleton objective; rotations and flips.
- Metrics: water overlap, boundary agreement, skeleton overlap, add/remove recall, harmful edits, enclosed-island area/counts and island-object recall.
- Uncertainty: paired bootstrap over rivers, per-river scores and threshold sensitivity. Two CNN training seeds per fold; no claims of architecture optimality or universal transfer.
- All target-derived spatial strata are evaluation-only. The model inputs contain no cleaned geometry, target-derived distance maps, coordinates, river identity, or edit polygons.

Large generated data and model files are excluded from Git by the local `.gitignore`. Code and the Markdown report are retained. Original source rasters on `/Volumes/PhD` must remain accessible to rebuild the samples.
