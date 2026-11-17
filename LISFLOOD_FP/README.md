# First custom 1D hydraulic prototype

Completed pilot: [example-56 report](results/example56/REPORT.md),
[hydrograph comparison](results/example56/hydrograph_comparison.png), and
[dynamic flow split](results/example56/dynamic_split.png).

Follow-up: [same-Manning-roughness comparison](results/example56/ROUGHNESS_COMPARISON.md)
uses `n=0.020446875`, the saved fixed-depth MC calibration value. Reproduce with
`run_example56.py --dx 100 --max-step 300 --n 0.020446875 --tag dx100_mc_roughness`,
then run `compare_roughness.py` to generate the comparison table and plot.

The [boundary calibration experiment](results/example56/boundary_calibration/REPORT.md)
calibrates the original network separately for normal flow, half/double outlet
friction slope, and an idealized 10 km downstream extension, then transfers each
fitted n to `S014_unit_10`, `S008_unit_6` and `S012_unit_12`. Run/resume with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python LISFLOOD_FP/run_boundary_experiment.py --workers 3 --stage all
MPLCONFIGDIR=/tmp/lisflood-prototype-mpl /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python LISFLOOD_FP/summarize_boundaries56.py
```

Roughness is fitted to the previously reviewed peak time with the prior soft
peak-height band; whole-window RMSE/NSE remain diagnostics. All collapse
parameters are frozen from the parent fit. The extension is evaluated at the
original outlet, with control-volume storage included in the flux reconstruction.

This is an independently implemented **diffusive-wave network model**, not the
LISFLOOD-FP executable or a reproduction of its numerical scheme. The hydraulic
approximation follows the continuity/friction/free-surface-gradient approach
described in [Bristol's model overview](https://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/structure/).

The agreed first-pass assumptions are rectangular channels, cached reconstructed
water-surface elevations used directly as a fixed bed proxy, constant Manning
roughness, no lateral runoff, no floodplain and no bed evolution. Depth and flow
allocation evolve. There is no local or convective acceleration, junction momentum
balance, or calibrated junction loss. This prototype is restricted to continuously
wet, gradually varying flows; it rejects dry accepted states and is not a general
wetting/drying or shock solver.

## Equations and numerical method

Each physical link is divided into segments no longer than the requested `dx`.
Length, width and endpoint elevations come from the input frame, not RAPID's
effective lengths, coefficients, reference depths or celerity caps. Beds interpolate
linearly along links. A shared stage exists at each junction.

For face `e=(i,j)`, with rectangular width `W`, depth `h=(h_i+h_j)/2` and
water-surface slope `s=((z_i+h_i)-(z_j+h_j))/dx`:

```
A = W h
R = A / (W + 2h)
K = A R^(2/3) / n
Q_ij = K s / (s^2 + epsilon^2)^(1/4), epsilon = 1e-10
C_i dh_i/dt = sum(Q_in) - sum(Q_out) + boundary_flux_i
C_i = sum(incident W dx / 2)
```

The smoothed signed square root allows reversals and zero internal flow at a
level water surface. At ordinary gradients it approaches `sign(s)*sqrt(abs(s))`.
Junction control volumes contain the incident half-segment storage, with no
additional arbitrary junction area. Consequently an instantaneous flow imbalance
at a junction equals the change of water stored in its finite control volume.
Refining the grid reduces this junction storage.

SciPy's implicit adaptive BDF integrator solves the coupled depth equations with
an analytic sparse Jacobian. Inflow and outflow volumes are integrated as extra
states, enabling a direct check of `delta_storage = integral(Qin-Qout)`.
An independent trapezoidal integral of the piecewise-linear forcing checks the
boundary-volume integration. The diagnostic output interval is five minutes;
the adaptive internal timestep is separate.

Normal depths seeded with width-based flow allocation are only an initial guess.
Constant initial inflow is then simulated in one-day chunks until every control
volume's flux residual is below `1e-8` of inlet discharge. Width fractions are
never used in the transient flux law. The outlet uses its unchanged terminal
width and bed slope in a normal-flow Manning rating.

## Example 56

`run_example56.py` compares the cached `S000_base` (46 links) with the existing
`S014_unit_10` collapse (42 links). Five original links between nodes 20 and 17
become one link; the other 41 links match exactly in topology, length, width and
endpoint elevation. The collapse was chosen for its clean unchanged exterior and
substantial branching section, before inspecting simulation outcomes.

These are historical local October hydraulic caches, **not a verification of the
currently selected production geometry**. The active master data drive was
unmounted. Input hashes and exact paths are saved with each run. The independent
pilot leaves all existing routing products unchanged.

The inflow is the cached observed discharge series from 21 June to 17 July 2025,
interpreted as point values with linear interpolation, followed by a 96-hour hold
at the last value. Gaps are counted in the manifest. Roughness `n=0.035` is shared
and uncalibrated. No observed downstream hydrograph is fitted.

## Reproduce

From the repository root, using the existing scientific Python environment:

```bash
/opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python -m unittest discover -s LISFLOOD_FP -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python LISFLOOD_FP/run_example56.py --dx 200 --max-step 300 --tag dx200_verified
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python LISFLOOD_FP/run_example56.py --dx 100 --max-step 300 --tag dx100
MPLCONFIGDIR=/tmp/lisflood-prototype-mpl /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python LISFLOOD_FP/summarize_example56.py
```

Dependencies: NumPy, SciPy, pandas, NetworkX, PyArrow, Matplotlib. Tests cover
steady normal flow, internal lake at rest, symmetric transient branch allocation,
mass conservation, reverse flow, vertical-datum invariance, conflicting input
heads and the analytic Jacobian against finite differences.

The report distinguishes numerical conservation/convergence from physical
validation. Agreement between mesh resolutions cannot validate the assumed bed,
roughness, rectangular sections or downstream boundary. The reported centroid
uses positive excess above a line connecting the initial and final observed
inflow, held constant during the tail; it is a descriptive metric specific to
this pilot, not an automatic replacement for the repository's established metrics.
