# Gauge means and discharge variability — 5 October 2026

Reference **HYDROGRAPH-MEAN-SPREAD-2026-10-05**. Requested after predictor residuals and example 37's competing peaks raised questions about lateral inflow, smoothing and long forcing windows. These are descriptive checks, not changes to primary metrics or production routing.

## Missing lateral inflow

Current runs route shared upstream forcing without observed lateral/tributary inflow time series. Network tributary geometry does not supply the missing observed water inputs. Missing input can contribute to downstream discharge and shape differences; timing, storage, hydraulic assumptions, sampling and gauge uncertainty prevent attributing all differences to this mechanism.

For each example, take finite native gauge observations within the configured gauge-review period and their overlapping temporal coverage. Remove duplicate timestamps. Calculate the requested arithmetic means independently on each station's observations:

mean Q = sum(Q_i)/N; difference = mean Q_down - mean Q_up; percentage difference = 100*(mean Q_down/mean Q_up - 1).

This is a same-clock-window comparison, without peak alignment. Native sample counts/cadences need not match. `gauge_means.csv` also reports median sampling step, maximum gap and a trapezoidal time-weighted mean over each station's actual retained sample span. Interpolation across gaps in that sensitivity statistic is not evidence of unobserved discharge. A finite-window mean difference is not a direct estimate of lateral inflow or an event-volume balance; travel-time and boundary storage effects matter. No t-test treats autocorrelated samples as independent replicates.

| example | upstream_mean_cms | downstream_mean_cms | mean_difference_pct |
| --- | --- | --- | --- |
| 02 | 541.53 | 949.86 | 75.4 |
| 03 | 1760.19 | 2058.42 | 16.94 |
| 05 | 500.82 | 546.15 | 9.05 |
| 11 | 2904.14 | 3416.58 | 17.65 |
| 13 | 694.83 | 1858.26 | 167.44 |
| 15 | 1410.63 | 2624.33 | 86.04 |
| 21 | 11584.55 | 13800.45 | 19.13 |
| 22 | 19318.6 | 24756.56 | 28.15 |
| 37 | 1805.37 | 3371.99 | 86.78 |
| 41 | 5743.91 | 7669.91 | 33.53 |
| 45 | 1149.48 | 1268.58 | 10.36 |
| 46 | 1040.19 | 1337.63 | 28.6 |
| 48 | 193.93 | 335.27 | 72.88 |
| 54 | 867.19 | 798.74 | -7.89 |
| 55 | 290.04 | 360.22 | 24.2 |
| 56 | 953.91 | 1011.77 | 6.07 |

Downstream arithmetic means exceed upstream means in **15/16** examples. The median relative difference is **26.17%**. Time-weighted means give the same 15/16 direction; the largest difference from the arithmetic percentage is **10.21 percentage points**. These examples are not assumed to be independent basins.

## Candidate discharge-SD response: initial example-37 check

Use population SD (ddof=0): sigma_Q = sqrt(sum((Q_i - mean Q)^2)/N). The candidate response is 100*(sigma_child/sigma_parent - 1); negative denotes reduced discharge variability within that window. CV = sigma_Q/mean Q is also exported. SD has units m³/s and is not temporal spread or maximum-peak attenuation.

Parent and children have identical time arrays and inflow values (asserted). Evaluate both the entire saved run and the shared gauge-review window. The latter is a sensitivity window, not a guarantee of capturing every delayed output feature.

| window | state | std_cms | std_change_vs_parent_pct |
| --- | --- | --- | --- |
| full_routed_event | forcing | 225.9889 | 0.0291 |
| full_routed_event | parent | 225.9232 | 0.0 |
| full_routed_event | island_37_272 | 225.9982 | 0.0332 |
| full_routed_event | CB001_bubble_3 | 226.201 | 0.123 |
| full_routed_event | S274_unit_280 | 226.0902 | 0.0739 |
| shared_gauge_window | forcing | 320.691 | -1.3067 |
| shared_gauge_window | parent | 324.937 | 0.0 |
| shared_gauge_window | island_37_272 | 325.0922 | 0.0478 |
| shared_gauge_window | CB001_bubble_3 | 323.0778 | -0.5722 |
| shared_gauge_window | S274_unit_280 | 325.1731 | 0.0727 |

The full compound has only **+0.123%** SD change over the complete run and **-0.572%** over the gauge window. Thus the visual increase in short-timescale fluctuations does not imply a large increase in global discharge SD. The event's broad amplitude/recession dominates this statistic, and timing/window boundaries can alter it. Do not adopt SD as a primary attenuation response based on this initial check. A future short-timescale roughness measure would require a specified common sampling interval and sensitivity to gauge noise; none is adopted here.

## Shorter events / Excel proposal — no implementation

Shorter inputs can reduce routing cost, approximately with simulated duration at fixed internal timestep and segmentation, but peak celerity/stability constraints may still dominate. First define the target event from observations, retain adequate antecedent/warm-up and trailing recession/settlement, and verify full-versus-cropped parent/child timing and shape. Particularly slow networks require longer buffers. Use one shared window/forcing for parent and children. Plot-only cropping does not reduce routing time. Do not select a window to improve predictor R².

Proposed review-workbook fields: target event/peak IDs, proposed start/end, warm-up and recession buffers, cadence/gaps, justification, review status, and full-versus-cropped verification. No Excel, forcing, routed output or current metric definition was changed.

Reproduce with `RAPID/investigations/hydrograph_mean_spread_audit.py`. Sources: current all-pair feature inventory, gauge-quality event periods, station observations and `shared-head-anchored-depth-signed-inflow-20261003` outlet files. Exports: `gauge_means.csv`, `example37_spread.csv`.

## Total-variation follow-up — 5 October 2026

Reference **HYDROGRAPH-TV-2026-10-05**, requested after SD failed to capture visible example-37 fluctuations. Reproduce with `RAPID/investigations/hydrograph_total_variation_audit.py`; results: `example37_total_variation.csv`. No forcing or primary response changed.

TV(Q) = sum(abs(Q[i+1]-Q[i])). Report 100*(TV_child/TV_parent-1). All parent/child time arrays and inputs are checked for equality; native saved cadence is 12.5833 minutes. TV is invariant to a constant discharge offset, but scales with event amplitude. For a separate amplitude-adjusted comparison define A = max(Q)-max(Q_start,Q_end) and V = TV/A. The existing peak-excess metric already uses an above-reference amplitude; total-discharge peak percentage has a larger denominator.

| Collapse | Full-event TV change | Full-event change in TV / own amplitude |
| --- | ---: | ---: |
| Island 272 | +0.40% | +0.37% |
| Unit 280 | +1.01% | +0.70% |
| Whole compound 3 | +17.48% | +14.03% |

The compound TV increase is +18.12% within the gauge window (+10.66% after each waveform's own amplitude normalization). This is a clearer roughness signal than SD. It does not mean global peak discharge increased by that amount, or that physical attenuation is fully described. Normalize both curves by a common parent amplitude for overlay plots; that common factor cancels out of the percentage TV ratio. Normalizing each curve by its own amplitude instead deliberately removes its amplitude change.

A supplementary descriptive quantity removes the minimal monotone rise/fall variation: excess TV = TV - (2*max(Q)-Q_start-Q_end). This is nonnegative for any sequence and equals zero for a monotone rise followed by monotone fall, including unequal endpoint levels. The export includes this exploratory quantity; no extra response is adopted.

Sampling sensitivity is material. Taking every 2nd, 4th or 8th sample from a common origin gives whole-compound TV increases of +16.58%, +13.54% and +9.41% (native: +17.48%). This is a decimation diagnostic, not an anti-aliased resampling prescription. Thus TV detects retained fluctuations but must be evaluated at a common, documented physical cadence before comparing networks; noise, window endpoints and other peaks also affect it. This initial check covers example 37 only, and does not yet establish cross-network validity.
