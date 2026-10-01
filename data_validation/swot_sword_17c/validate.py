"""Reproduce the local SWOT/SWORD coverage comparison without modifying source data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json, hashlib, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SWOT = Path('/Volumes/PhD/SWOT/RiverSP_D_parq/node')
SWORD = Path('/Volumes/PhD/SWORD/v17c')
SUMMARY = ROOT/'SWORD_gauge_match/outputs/hierarchy_example_reach_summary.csv'
FILTER_SOURCE = Path('/Users/6256481/Code/river-morphology-atlas/open_SWOT_files.py')

def main():
    start = time.time()
    summary = pd.read_csv(SUMMARY)
    associations = []
    active = {int(p.name.split('_')[0]) for p in Path('/Volumes/PhD/river_hierarchy/network_variants/output').glob('[0-9][0-9]_indep__anchored')}
    for row in summary.itertuples():
        ids = set()
        for field in ['reaches_between','upstream_reach_ids','downstream_reach_ids']:
            v = getattr(row,field)
            if pd.notna(v): ids.update(int(float(x)) for x in str(v).split(';') if x)
        for rid in sorted(ids): associations.append((row.example_id,row.sword_region.upper(),rid,row.example_id in active))
    assoc = pd.DataFrame(associations,columns=['example_id','region','reach_id','generated_network'])
    assoc.to_csv(OUT/'example_reaches.csv',index=False)
    reaches = sorted(assoc.reach_id.unique())
    sword_parts = []
    source_files = []
    for region in sorted(assoc.region.unique()):
        p = SWORD/f'sword_{region}_v17c_nodes.parquet'
        d = pq.read_table(p,columns=['node_id','reach_id','x','y','dist_out','wse_obs_p50','n_obs'],filters=[('reach_id','in',reaches)]).to_pandas()
        d['region'] = region
        sword_parts.append(d)
        source_files.append({'path':str(p),'size_bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns})
    sword = pd.concat(sword_parts,ignore_index=True)
    assert not sword.duplicated(['node_id','reach_id']).any()
    sword['sword_p50'] = sword.wse_obs_p50.where(np.isfinite(sword.wse_obs_p50) & (sword.wse_obs_p50 > -1e10))
    missing_reaches = assoc.loc[~assoc.reach_id.isin(sword.reach_id)]
    missing_reaches.to_csv(OUT/'reaches_absent_from_sword.csv',index=False)
    paths = sorted(p for p in SWOT.glob('*.parquet') if not p.name.startswith('.'))
    cols = ['node_id','reach_id','wse_sm','wse_sm_u','dark_frac','xtrk_dist','time_str','cycle','pass','crid','version','granule']
    def scan(p):
        audit = {'file':p.name,'size_bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
        try:
            f = pq.ParquetFile(p)
            audit['file_rows'] = f.metadata.num_rows
            typ = f.schema_arrow.field('reach_id').type
            vals = [str(x) for x in reaches] if pa.types.is_string(typ) else reaches
            present = [c for c in cols if c in f.schema_arrow.names]
            required = set(cols[:7])
            if not required.issubset(present): raise ValueError(f'Missing fields: {required-set(present)}')
            d = pq.read_table(p,columns=present,filters=[('reach_id','in',vals)],use_threads=False).to_pandas()
            audit['target_rows'] = len(d)
            audit['status'] = 'ok'
            if not d.empty:
                d['source_file'] = p.name
                for c in ['node_id','reach_id']: d[c] = pd.to_numeric(d[c],errors='raise').astype('int64')
            return audit,d
        except Exception as e:
            audit.update(status='error',error=str(e)); return audit,None
    audits,parts = [],[]
    print(f'SWORD: {len(sword):,} nodes; {len(summary)} examples; SWOT: {len(paths):,} files',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,(audit,d) in enumerate(pool.map(scan,paths),1):
            audits.append(audit)
            if d is not None and not d.empty: parts.append(d)
            if i%500 == 0: print(f'{i}/{len(paths)} files; {sum(len(x) for x in parts):,} target rows; {time.time()-start:.0f}s',flush=True)
    pd.DataFrame(audits).to_csv(OUT/'file_manifest.csv',index=False)
    if any(x['status']=='error' for x in audits): raise RuntimeError('Some files failed: see file_manifest.csv')
    obs = pd.concat(parts,ignore_index=True)
    # Exact repository predicate; a second flag checks whether fill values in QC fields slip through.
    obs['passes_repository_filter'] = ((obs.wse_sm_u<=1)&(obs.dark_frac<=.5)&obs.xtrk_dist.abs().between(10000,60000)&(obs.wse_sm>0))
    obs['passes_valid_qc_filter'] = obs.passes_repository_filter & np.isfinite(obs[['wse_sm','wse_sm_u','dark_frac','xtrk_dist']]).all(axis=1) & (obs.wse_sm_u>=0)&(obs.dark_frac>=0)
    obs.to_parquet(OUT/'target_observations.parquet',index=False)
    raw = obs.groupby(['node_id','reach_id']).size().rename('swot_rows_before_filter')
    good = obs.loc[obs.passes_repository_filter].copy()
    stats = good.groupby(['node_id','reach_id']).agg(swot_p50=('wse_sm','median'),swot_observations=('wse_sm','size'),first_observation=('time_str','min'),last_observation=('time_str','max'))
    strict = obs.loc[obs.passes_valid_qc_filter].groupby(['node_id','reach_id']).wse_sm.median().rename('swot_p50_valid_qc')
    nodes = sword.merge(raw,on=['node_id','reach_id'],how='left',validate='one_to_one').merge(stats,on=['node_id','reach_id'],how='left',validate='one_to_one').merge(strict,on=['node_id','reach_id'],how='left',validate='one_to_one')
    for c in ['swot_rows_before_filter','swot_observations']: nodes[c] = nodes[c].fillna(0).astype(int)
    nodes['sword_valid'] = nodes.sword_p50.notna()
    nodes['swot_valid'] = nodes.swot_p50.notna()
    nodes['coverage_category'] = np.select([nodes.sword_valid & nodes.swot_valid,~nodes.sword_valid & nodes.swot_valid,nodes.sword_valid & ~nodes.swot_valid],['both','swot_only','sword_only'],default='neither')
    nodes['swot_minus_sword_m'] = nodes.swot_p50-nodes.sword_p50
    nodes.to_csv(OUT/'unique_node_comparison.csv',index=False)
    by_example = assoc.merge(nodes,on=['region','reach_id'],how='left',validate='many_to_many')
    by_example.to_csv(OUT/'example_node_comparison.csv',index=False)
    extras = raw.reset_index().merge(sword[['node_id','reach_id']],on=['node_id','reach_id'],how='left',indicator=True)
    extras = extras.loc[extras['_merge']=='left_only'].drop(columns='_merge').merge(stats,on=['node_id','reach_id'],how='left')
    extras.to_csv(OUT/'swot_nodes_absent_from_sword.csv',index=False)
    def metrics(d):
        return dict(sword_nodes=len(d),sword_valid=int(d.sword_valid.sum()),swot_valid=int(d.swot_valid.sum()),sword_missing=int((~d.sword_valid).sum()),swot_missing=int((~d.swot_valid).sum()),recovered_by_swot=int(d.coverage_category.eq('swot_only').sum()),sword_only=int(d.coverage_category.eq('sword_only').sum()),neither=int(d.coverage_category.eq('neither').sum()),net_gain=int(d.swot_valid.sum()-d.sword_valid.sum()),median_swot_minus_sword_m=float(d.swot_minus_sword_m.median()),median_absolute_difference_m=float(d.swot_minus_sword_m.abs().median()))
    example_metrics = pd.DataFrame([dict(example_id=int(e),generated_network=bool(d.generated_network.iloc[0]),**metrics(d)) for e,d in by_example.groupby('example_id')])
    example_metrics.to_csv(OUT/'example_summary.csv',index=False)
    dupkey = [x for x in ['node_id','reach_id','cycle','pass','time_str'] if x in good.columns]
    dedup = good.drop_duplicates(dupkey).groupby(['node_id','reach_id']).wse_sm.median()
    p50diff = stats.swot_p50.sub(dedup).abs()
    meta = dict(sword_sources=source_files,example_source=str(SUMMARY),example_source_sha256=hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),filter_source=str(FILTER_SOURCE),filter_source_sha256=hashlib.sha256(FILTER_SOURCE.read_bytes()).hexdigest(),filter='wse_sm_u <= 1 AND dark_frac <= 0.5 AND ABS(xtrk_dist) BETWEEN 10000 AND 60000 AND wse_sm > 0',sword_field='wse_obs_p50',swot_field='wse_sm',scope='All nodes on saved example reaches, including endpoint reaches; no optional reach buffer or spatial crop; no WSE filling.',files_scanned=len(paths),file_errors=0,total_archive_rows=int(sum(a['file_rows'] for a in audits)),target_rows=len(obs),filtered_rows=len(good),invalid_qc_rows_passing_repository_filter=int((obs.passes_repository_filter & ~obs.passes_valid_qc_filter).sum()),strict_qc_coverage_changed_nodes=int((nodes.swot_valid != nodes.swot_p50_valid_qc.notna()).sum()),duplicate_observation_rows=int(good.duplicated(dupkey).sum()),nodes_p50_changed_by_dedup=int((p50diff>1e-9).sum()),swot_extra_nodes=len(extras),swot_extra_valid_nodes=int(extras.swot_p50.notna().sum()),first_filtered_observation=good.time_str.min(),last_filtered_observation=good.time_str.max(),all_examples=metrics(nodes),generated_examples=metrics(nodes.loc[nodes.reach_id.isin(assoc.loc[assoc.generated_network,'reach_id'])]),elapsed_seconds=time.time()-start)
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)
    print(example_metrics.to_string(index=False),flush=True)

if __name__=='__main__': main()
