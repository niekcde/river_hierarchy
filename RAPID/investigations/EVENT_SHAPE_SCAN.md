# Preliminary event-shape scan

Run `scan_event_shape.py` using the river-hierarchy-rivgraph environment.
Outputs: `RAPID/assets/event_shape_preliminary/`.

Scope: available active baseline and variant hydrographs for the example IDs in
rapid_hydrograph_analysis. Island aliases without their own hydrograph paths are
not counted again. Hydrographs can exist even when registry scalar metrics are
missing. No routing is rerun. Compound edge exclusions are not applied.

The scan removes initial constant-forcing spin-up, then compares the parent and
variant from a shared UTC start through each simulation’s own end. A shared end would
truncate slower parents because warm-down lengths differ by state.
The first forcing change is detected at 1e-6 of the forcing range. Whole records
may include several input pulses. Counts are NOT attribution of peaks to branches.

Peak count thresholds are 1%, 5%, and 10% of the parent's dominant peak
prominence, using scipy.signal.find_peaks. Both curves use the same discharge
threshold within each pair. Dominant-peak widths are measured at 50% and 95%
of prominence below the peak (scipy.signal.peak_widths); they do not measure
the duration of separated later pulses. Prominence is measured relative to
surrounding minima, not a physical baseflow separation. Changes in prominence
must not be presented as definitive event-excess attenuation.

This deliberately avoids deriving flood-volume metrics from an unvalidated
background separation in real events. Recommended next stage: define shared
forcing event windows and background treatment, check event/tail completeness,
then calculate normalized event-volume mean time, temporal standard deviation,
and cumulative-volume duration (t99-t01), with sensitivity to separation and
window choices. Treat incomplete recessions as censored. Report collapse changes
as percentages of the parent metric; centroid translation should be normalized
by a parent duration or compared through centroid lag, not by a UTC timestamp.
