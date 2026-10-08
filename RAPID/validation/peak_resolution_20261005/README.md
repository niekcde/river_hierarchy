# Peak resolution audit — 5 October 2026

Reference: RAPID-PEAK-RESOLUTION-2026-10-05. Requested after the refreshed
analysis showed 2,415/3,517 comparisons with zero sampled peak-time shift.

Production stores forcing-interval mean discharge (550–1540 seconds across
cases). hydrograph.py selects a dominant peak and assigns the corresponding
saved timestamp without sub-sample peak interpolation. Parent/child equality
of those timestamps does not imply identical hydrographs or zero travel time.

The targeted investigation selects examples 03, 15 and 46; in each, two
zero-shift children (median and largest absolute centroid difference) plus the
smallest nonzero sampled-shift control. Fingerprints are deduplicated. Including
bases, 12 geometries were routed at production and half-production clocks:
24 runs, retaining outlet values at every native substep. Three-point parabolic
peak estimates and discrete native peaks are both retained in runs.csv.

All six selected zero pairs resolve into shifts: -2.419, -4.376, -3.708,
+5.481, -7.744, -12.883 minutes. Changes on halving the timestep are below
0.05 seconds across all nine pairs. This is numerical convergence under the
same held forcing, not observational timing accuracy or proof for all zero rows.
Interpolation alone on the coarse existing output is within 6.34 seconds of
the half-clock estimate in this pilot. The coarse nonzero controls can also
misrepresent magnitude: discrete-bin changes need not approximate the refined
shift closely.

The cascade uses the same Muskingum recurrence and steady initialization.
Every routed segment is checked for finite/nonnegative native flow. Native
outputs averaged over original intervals reproduce production within 1e-6
relative plus 1e-5 m3/s absolute tolerance; small differences include serialized
production input precision. Sparse-engine initial probes match within 1e-9
relative plus 1e-7 m3/s. Regression tests verify native-output aggregation and
split/rejoin agreement for two fractional clocks. No production run, scientific
parameter or primary notebook metric was changed by this diagnostic.

Review comparisons.csv and peak_closeups.png. Reproduce with
`python RAPID/investigations/peak_resolution_check.py` using the repository
Python environment and the production results volume mounted.

Next: validate a guarded interpolated peak extractor on broad, tied and multiple
peaks before adoption in both real and synthetic analysis. Retain sampled peaks
and resolution flags, and do not equate interpolation digits with measured
precision. This small pilot does not establish the full-population zero fraction
after refined extraction.
