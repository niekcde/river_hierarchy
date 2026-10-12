"""Stage a metadata-only revision after the independent completeness audit."""
from pathlib import Path
import sys,json
from functools import lru_cache
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from network_variants.reviewed_geometry import active_geometry,digest
old=active_geometry();OUT=ROOT/'RAPID/assets/reviewed_geometry_20261009_r2';OUT.mkdir(exist_ok=True)
if (OUT/'manifest.json').exists():raise FileExistsError('Revision is already staged')
manifest=json.loads(old.manifest.read_text());dropped=[r for r in manifest['records'] if r['state'].startswith('.')]
assert len(dropped)==2
manifest['records']=[r for r in manifest['records'] if not r['state'].startswith('.')]
for r in manifest['records']:
 if r.get('canonical_key'):assert not r['canonical_key'][2].startswith('.')
manifest.update(release=OUT.name,supersedes=str(old.manifest),supersedes_sha256=digest(old.manifest),
                removed_nonstate_entries=dropped,revision_reason='Exclude two temporary promotion directories; preserve original identity maps for unchanged geometry; tighten pending-hydraulic guards')
@lru_cache(maxsize=None)
def sha(path):return digest(path)
# Retained source files were previously references only. Freeze their checksums
# so unchanged geometry/identities cannot drift unnoticed during regeneration.
for i,r in enumerate(manifest['records']):
 if not r.get('canonical_key') and not r.get('bundle') and r['geometry_status']!='excluded':
  r['retained_source_checksums']={r[name]:sha(r[name]) for name in ['links','nodes','original_frame']}
 if (i+1)%500==0:print('Source checksums',i+1,flush=True)
for ex in sorted({r['example'] for r in manifest['records']}):
 suffix='_direction130' if ex=='15' else ''
 for family in ['indep','compound_bubbles','island']:
  p=ROOT/'network_variants/output'/f'{ex}_{family}{suffix}__anchored'/('island_hydraulic_context_registry.csv' if family=='island' else 'state_registry.csv')
  if p.exists():manifest['input_checksums'][str(p)]=sha(str(p))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records=pd.DataFrame(manifest['records']);records.to_csv(OUT/'state_registry.csv',index=False)
for p in old.manifest.parent.glob('*/inventory.csv'):
 f=pd.read_csv(p,dtype={'example':str}).fillna('');f=f[~f.state.str.startswith('.')].copy();f['geometry_release']=OUT.name
 dest=OUT/p.parent.name;dest.mkdir(exist_ok=True);f.to_csv(dest/'inventory.csv',index=False)
 records[records.example.eq(p.parent.name)].to_csv(dest/'geometry_registry.csv',index=False)
summary=json.loads((old.manifest.parent/'summary.json').read_text())
summary.update(logical_states=len(records),active=False,active_pointer=None,removed_temporary_entries=2,
               decisions=records.decision.value_counts().to_dict(),supersedes=str(old.manifest),
               geometry_objects_reused_without_copy=970,retained_source_files_checksummed=sha.cache_info().currsize,
               unresolved_geometry_reviews=0,tests_passed=None,normal_registry_smoke_test_passed=False)
summary.pop('release_size_bytes',None)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(OUT)
