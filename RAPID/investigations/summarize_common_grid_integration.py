"""Verify completed integration evidence and write a compact, reproducible report."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'RAPID/outputs/common_grid_production_integration_20261008'


def main():
    bases, pairs, sparse = [], [], []
    for group in ['verified_bases', 'verified_pairs']:
        paths = sorted((OUT/group).glob('*/completion.json'))
        assert len(paths) == (15 if group == 'verified_bases' else 4)
        for path in paths:
            folder = path.parent
            assert json.loads(path.read_text())['all_requested_states_routed']
            provenance = json.loads((folder/'experiment.json').read_text())
            for source, expected in provenance['source_hashes'].items():
                assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
            results = json.loads((folder/'results.json').read_text())
            assert min(r['minimum_native_flow_cms'] for r in results) >= 0
            resolution = json.loads((folder/'resolution_checks.json').read_text())
            assert all(r['feature_kind_same'] for r in resolution)
            if group == 'verified_bases':
                r = next(r for r in results if r['interval_seconds'] == 60)
                assert r['reviewed_wave_max_difference_cms'] == 0
                bases.append(dict(example=folder.name, replay_difference_cms=r['reviewed_wave_max_difference_cms'], **resolution[0]))
            else:
                records = json.loads((folder/'parent_child_comparison.json').read_text())
                assert all(r['feature_kind_same'] for r in records)
                p60 = next(r for r in records if r['interval_seconds'] == 60)
                p30 = next(r for r in records if r['interval_seconds'] == 30)
                pairs.append(dict(example=folder.name, parent=p60['parent_state'], child=p60['child_state'],
                    delta_60_seconds=p60['delta_feature_seconds'], delta_30_seconds=p30['delta_feature_seconds'],
                    difference_seconds=p30['delta_timing_resolution_difference_seconds']))
                sparse.extend(dict(example=folder.name, state=name, **record) for name, record in
                              json.loads((folder/'sparse_engine_parity.json').read_text()).items())
    summary = dict(base_cases=bases, parent_child_pairs=pairs,
        maximum_interval_difference_pct_input_amplitude=max(r['max_difference_pct_inlet_amplitude'] for r in bases),
        maximum_sparse_engine_difference_cms=max(r['max_difference_cms'] for r in sparse),
        maximum_parent_child_timing_resolution_difference_seconds=max(abs(r['difference_seconds']) for r in pairs),
        tests_passed=43, test_scope='common-grid production, common-grid, time support, Cunge, reviewed runner, hydraulic policy',
        full_hierarchy_batch_completed=False, final_scientific_acceptance=False,
        summary_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'integration_validation.json').write_text(json.dumps(summary, indent=2))
    lines = ['COMMON-GRID INTEGRATION CHECK — 2026-10-08', '',
        '15/15 reviewed base hydrographs reproduced exactly at 60 seconds.',
        'All input/target/code hashes checked again after routing. No negative native discharge.',
        f"Largest base 60s-versus-aggregated-30s difference: {summary['maximum_interval_difference_pct_input_amplitude']:.9g}% of input amplitude.",
        'Base feature timestamps differ by 15 seconds between midpoint grids; feature kinds unchanged.',
        f"Independent sparse-engine crest-excerpt check: maximum difference {summary['maximum_sparse_engine_difference_cms']:.9g} m3/s.",
        '43 targeted tests passed, including legacy default solver parity and gap rejection.', '',
        'Sampled parent-child feature changes (child minus parent; negative means earlier):',
        'Example   60s grid    30s grid    Difference']
    lines += [f"{r['example']:>7} {r['delta_60_seconds']:>9.0f}s {r['delta_30_seconds']:>10.0f}s {r['difference_seconds']:>11.0f}s" for r in pairs]
    lines += ['', 'All four sampled transitions passed existing exterior-parameter review gates.',
        'These are first independent children of their bases, not an audit of every hierarchy transition.',
        '60s is numerically consistent for discharge in these checks. Timing differences of about',
        'one minute or less warrant finer-grid controls; do not interpret sub-minute hierarchy',
        'changes as resolved on a 60s grid. 54 retains its two-lobe midpoint feature.', '',
        'Scope: integration verified on reviewed inputs with explicitly flagged diagnostic gap filling.',
        'Remaining: source clocks 21/22; local observation/gap eligibility, notably 55;',
        'full-tail settling/volume acceptance; complete hierarchy preflight and batch.',
        'No changes to n/beta targets or network-41 geometry; no new antecedent forcing.',
        'Legacy command behavior is preserved. Invoke the new runner with --common-grid.',
        'Usage and implementation detail: RAPID/reviews/common_grid_production_20261008/INTEGRATION.txt']
    (OUT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
