"""Read-only source audit and compact migration inventory; never routes a network."""
from pathlib import Path
import hashlib
import json
import lzma

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'RAPID/outputs/temporal_and_target_audit_20261008'
OUT = ROOT / 'RAPID/outputs/temporal_policy_review_20261008'


def main():
    OUT.mkdir(exist_ok=True)
    clocks = pd.read_csv(AUDIT / 'all_calibration_clock_audit.csv', dtype={'example': str})
    decisions = clocks[['example', 'method', 'old_error_min', 'native_error_min',
                        'tolerance_min', 'old_within_tolerance', 'native_within_tolerance']].copy()
    def action(row):
        if row.example == '11':
            return 'remain_excluded'
        if row.example == '54':
            return 'resolve_two_crest_objective_before_fit'
        if row.example == '03':
            return 'resolve_plateau_metric_before_fit'
        if not row.native_within_tolerance:
            return 'priority_recalibration_candidate_after_feature_review'
        return 'retain_pending_support_and_feature_validation'
    decisions['action'] = decisions.apply(action, axis=1)
    decisions['hierarchy_rerun'] = 'conditional_on_changed_dynamics; otherwise missing_native_diagnostics_only'
    decisions.to_csv(OUT / 'example_decisions.csv', index=False)

    book = load_workbook(ROOT / 'SWORD_gauge_match/outputs/mask_hydro_quality.xlsx',
                         read_only=True, data_only=True)
    rows = list(book['Hydrograph_quality'].values)
    book.close()
    quality = pd.DataFrame(rows[1:], columns=rows[0])
    records, headers, mrc_evidence = [], {}, {}
    for ex in sorted(clocks.example.unique()):
        row = quality.loc[quality.Example == int(ex)].iloc[0]
        for role, field in [('upstream', 'Up'), ('downstream', 'Down')]:
            key = row[field]
            source = ROOT / f'SWORD_gauge_match/outputs/subdaily_values/{key.split(":")[0]}/subdaily_timeseries.parquet'
            cols = ['provider', 'provider_station_id', 'provider_series_name', 'provider_series_id']
            metadata = pd.read_parquet(source, filters=[('station_key', '==', key)], columns=cols).drop_duplicates()
            record = dict(example=ex, role=role, station_key=key, source=str(source),
                          provider_metadata=metadata.to_dict('records'),
                          support='point_provisional', interval_bounds_available=False)
            if key.startswith('CA:'):
                evidence = []
                for station in metadata.provider_station_id.unique():
                    for path in sorted((source.parent / 'manual_download').glob(f'*@{station}.*.xz')):
                        lines = []
                        with lzma.open(path, 'rt') as handle:
                            for line in handle:
                                lines.append(line.rstrip())
                                if not line.startswith('#'):
                                    break
                        headers[str(path)] = lines
                        evidence.extend(lines)
                if any('Interpolation type: Instantaneous Values' in s for s in evidence):
                    record['support'] = 'point_confirmed_in_archive_header'
                record['clock_evidence'] = 'archive ISO 8601 UTC column; importer uses that column'
            elif key.startswith(('TH:', 'LA:', 'KH:')):
                record['clock_evidence'] = 'unresolved: header UTC+07:00 but row strings end Z; do not shift automatically'
                record['support'] = 'rating_curve_of_stage; stage_support_needs_confirmation'
                for path in sorted((source.parent / 'Waterlevel').glob('*.csv')):
                    with path.open() as handle:
                        mrc_evidence[str(path)] = [next(handle).rstrip(), next(handle).rstrip()]
            elif key.startswith('BR:'):
                record['clock_evidence'] = 'importer parses DataHora as UTC; verify timezone of original ANA payload'
            elif key.startswith('US:'):
                record['support'] = 'point_supported_by_USGS_continuous_product'
            elif key.startswith('GF:'):
                record['support'] = 'point_supported_by_observations_tr_Q_product'
            elif key.startswith('CL:'):
                record['support'] = 'point_supported_by_instantaneo_import'
            records.append(record)
    (OUT / 'provider_support.json').write_text(json.dumps(records, indent=2))
    (OUT / 'canada_archive_headers.json').write_text(json.dumps(headers, indent=2))
    (OUT / 'mrc_clock_evidence.json').write_text(json.dumps(mrc_evidence, indent=2))
    sources = [AUDIT / 'all_calibration_clock_audit.csv', AUDIT / 'provenance.json',
               ROOT / 'SWORD_gauge_match/outputs/mask_hydro_quality.xlsx', Path(__file__)]
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        production_modified=False, routing_runs=0,
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        metadata_scope='Current station metadata plus Canadian raw archive headers; not a full raw-value reimport'), indent=2))
    print(decisions.to_string(index=False))
    print(f'Wrote {len(records)} station-role records and {len(headers)} archive headers to {OUT}')


if __name__ == '__main__':
    main()
