"""Refine selected-event lag fits and report context-dependent roughness ranges.

The literature ranges are interpretation screens, not automatic acceptance of
physical roughness. No geometry, gauge peak definition or production file changes.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

import travel_time_calibration as cal

SEARCH_BOUNDS = (.010, .100)
COMMON_BAND = (.025, .060)
OUTPUT = cal.ROOT/'RAPID/outputs/travel_time_calibration_refined'
SOURCES = {
    'natural_channels': 'https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/latest/basic-data-requirements/geometric-data/energy-loss-coefficients',
    'alluvial_beds': 'https://www.hec.usace.army.mil/confluence/rasdocs/rasum/6.2/performing-a-1d-unsteady-flow-analysis/calibration-of-unsteady-flow-models/roughness-coefficients',
    'sand_bed_base_values': 'https://www.fhwa.dot.gov/engineering/hydraulics/pubs/hif12004.pdf',
    'selection_method': 'https://www.usgs.gov/publications/guide-selecting-mannings-roughness-coefficients-natural-channels-and-flood-plains',
}


def roughness_context(n):
    """Interpretation only: no site bed-form or obstruction evidence is assumed."""
    if not np.isfinite(n) or n <= 0:
        raise ValueError('Manning n must be finite and positive')
    if n < .012:
        return 'Very low: requires specific smooth sand-bed evidence; check model compensation'
    if n < COMMON_BAND[0]:
        return 'Low: overlaps smoother alluvial-bed guidance; site evidence needed'
    if n <= COMMON_BAND[1]:
        return 'Within common screening band; physical roughness still unverified'
    return 'High: requires roughness/vegetation/storage explanation; check model compensation'


def expanded_case(original, bounds=SEARCH_BOUNDS):
    """Recheck the shared clock for the enlarged range; preserve all other inputs."""
    if not 0 < bounds[0] <= cal.BASE_N <= bounds[1]:
        raise ValueError('Search bounds must contain the positive baseline n')
    case = dict(original)
    fastest = cal.candidate_frame(case['frame'],case['reference'],bounds[0])
    _, upper = cal.compute_dt_bounds(fastest.rapid_k,x_value=.1,coefficient_policy='signed_current_inflow')
    dt_limit = min(original['routing_dt_s'], .99*upper)
    nsub = max(original['nsub'],int(math.ceil(case['forcing_step_s']/dt_limit)))
    case.update(bounds=bounds,nsub=nsub,routing_dt_s=case['forcing_step_s']/nsub)
    signature = dict(original_signature=original['signature'],bounds=bounds,nsub=nsub,
                     refinement_code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    case['signature'] = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    return case


def refine_case(case, previous_trials, output=OUTPUT, max_iterations=14, n_tolerance=1e-7):
    """Bracket the target and bisect until the sampled-output timing floor.

Evaluate all trial points on one new common clock, including comparisons within
the 0.025–0.060 interpretation band. Do not reuse old-clock scores as new trials.
"""
    if not case['target']['target_usable']:
        raise ValueError('Observed target is not usable for automatic calibration')
    prior = previous_trials.sort_values('n_manning').reset_index(drop=True)
    previous = prior.assign(distance=abs(prior.n_manning-cal.BASE_N)).sort_values(['abs_error_h','distance']).iloc[0]
    previous_n = float(previous.n_manning)
    rows = {}
    def evaluate(n):
        n = float(n)
        if not case['bounds'][0] <= n <= case['bounds'][1]:
            raise ValueError('Refinement candidate outside checked clock bounds')
        for prior_n in rows:
            if np.isclose(n,prior_n,rtol=0,atol=1e-12):
                return rows[prior_n]
        result, _ = cal.run_trial(case,n,output)
        rows[n] = result
        return result

    # Existing best and neighbours preserve local event/peak context. All n
    # values are rerouted or reused only under the NEW exact trial identity.
    seeds = [previous_n, cal.BASE_N, *COMMON_BAND]
    below = prior.loc[prior.n_manning < previous_n-1e-12,'n_manning']
    above = prior.loc[prior.n_manning > previous_n+1e-12,'n_manning']
    if len(below): seeds.append(float(below.max()))
    if len(above): seeds.append(float(above.min()))
    if np.isclose(previous_n,prior.n_manning.min()): seeds.append(case['bounds'][0])
    if np.isclose(previous_n,prior.n_manning.max()): seeds.append(case['bounds'][1])
    for n in sorted(set(seeds)):
        evaluate(n)

    def ranked(bounds):
        return sorted((r for n,r in rows.items() if bounds[0] <= n <= bounds[1]),
                      key=lambda r:(r['abs_error_h'],abs(r['n_manning']-previous_n)))

    stop_h = case['forcing_step_s']/7200. + 1e-12
    def solve(bounds):
        # Expand only when needed to bracket; boundaries are finite diagnostic
        # limits, never a claim that all values in them are physically acceptable.
        candidates = ranked(bounds)
        signs = [r['signed_error_h'] for r in candidates]
        if min(signs) > 0: evaluate(bounds[0])
        if max(signs) < 0: evaluate(bounds[1])
        iterations = 0
        reason = 'maximum iterations reached'
        for _ in range(max_iterations):
            ordered = sorted(ranked(bounds),key=lambda r:r['n_manning'])
            best = ranked(bounds)[0]
            if best['abs_error_h'] <= stop_h:
                reason = 'within half a forcing interval'
                break
            crossing = [(a,b) for a,b in zip(ordered[:-1],ordered[1:])
                        if a['signed_error_h']*b['signed_error_h'] < 0
                        and b['n_manning']-a['n_manning'] > n_tolerance]
            if not crossing:
                reason = 'no unresolved bracketed crossing in searched range'
                break
            a,b = min(crossing,key=lambda pair:min(pair[0]['abs_error_h'],pair[1]['abs_error_h']))
            evaluate((a['n_manning']+b['n_manning'])/2)
            iterations += 1
        return ranked(bounds)[0],reason,iterations

    best,stop_reason,iterations = solve(case['bounds'])
    common,common_reason,common_iterations = solve(COMMON_BAND)
    # The comparison-band search may also find a better unrestricted point.
    best = ranked(case['bounds'])[0]
    if best['abs_error_h'] <= stop_h: stop_reason = 'within half a forcing interval'
    trials = pd.DataFrame(rows.values()).sort_values('n_manning').reset_index(drop=True)
    same_clock_previous = rows[previous_n]
    acceptable_timing = trials.loc[trials.abs_error_h <= best['timing_resolution_h']]
    summary = dict(best)
    summary.update(
        previous_n=previous_n,previous_abs_error_h=float(previous.abs_error_h),
        previous_candidate_new_clock_error_h=same_clock_previous['abs_error_h'],
        search_lower=case['bounds'][0],search_upper=case['bounds'][1],
        best_at_search_boundary=bool(np.isclose(best['n_manning'],case['bounds'][0],atol=1e-10,rtol=0)
                                    or np.isclose(best['n_manning'],case['bounds'][1],atol=1e-10,rtol=0)),
        search_stop=stop_reason,refinement_iterations=iterations,
        numerical_target_tolerance_h=stop_h,
        within_observation_sampling_tolerance=bool(best['abs_error_h']<=best['timing_resolution_h']),
        roughness_context=roughness_context(best['n_manning']),
        site_roughness_verified=False,
        common_band_lower=COMMON_BAND[0],common_band_upper=COMMON_BAND[1],
        common_band_best_n=common['n_manning'],common_band_abs_error_h=common['abs_error_h'],
        common_band_search_stop=common_reason,common_band_iterations=common_iterations,
        common_band_within_sampling_tolerance=bool(common['abs_error_h']<=best['timing_resolution_h']),
        tested_n_within_sampling_tolerance=acceptable_timing.n_manning.tolist(),
        significant_forcing_peak_offset=bool(abs(best['forcing_minus_observed_upstream_peak_h'])>best['timing_resolution_h']),
        sampled_lag_nonmonotonic=bool((np.diff(trials.simulated_lag_h)<-case['forcing_step_s']/3600-1e-10).any()),
    )
    dest = Path(output)/case['example']/case['signature'][:20]
    dest.mkdir(parents=True,exist_ok=True)
    trials.to_csv(dest/'refinement_trials.csv',index=False)
    (dest/'refined_candidate.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return trials,summary
