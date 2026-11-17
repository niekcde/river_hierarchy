"""Read-only verification of the migrated current synthetic run."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'RAPID/src',ROOT/'synthetic_experiments/src'):sys.path.insert(0,str(p))
from synthetic_experiments.depth_pipeline import provenance
from synthetic_experiments.networks import digest
output=Path(sys.argv[1]);assert output.is_mount(),'Container not mounted at output path'
audit=json.loads((ROOT/'synthetic_experiments/outputs/storage_audit_20261027/audit.json').read_text())
manifest=json.loads((output/'manifest.json').read_text());assert manifest['source']==provenance()
jobs={j['id']:j for j in manifest['jobs']};count=0
for folder in (output/'groups').iterdir():
 assert folder.is_dir()
 status=json.loads((folder/'completion.json').read_text());job=jobs[folder.name]
 expected=digest(dict(job=job,source=manifest['source'],convergence=manifest['config']['convergence']))
 assert status['identity']==expected,(folder,'identity mismatch')
 for name in ['responses.csv','state_metrics.csv','hydrographs.parquet','parameters.csv']:
  assert (folder/name).is_file(),(folder,name)
 count+=1
assert count==audit['complete_with_required_files']
seen={}
with (output/'storage_migration_sha256.jsonl').open() as stream:
 for line in stream:
  entry=json.loads(line);seen[entry['path']]=entry
assert len(seen)==sum(v['files'] for v in audit['by_type'].values())
for rel,entry in seen.items():
 p=output/rel;assert p.is_file() and p.stat().st_size==entry['bytes'],rel
print(f'PASS: {len(seen):,} audited files present; {count:,} complete groups match original identities; source unchanged.')
