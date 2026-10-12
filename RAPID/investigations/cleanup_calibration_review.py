"""Remove only audited redundant output paths after canonical overview verification."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'RAPID/outputs/calibration_review'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 manifest=json.loads((OUT/'overview_manifest.json').read_text())
 assert manifest['serialized_values_verified'] and sha(OUT/'15_network_overview.html')==manifest['html_sha256']
 plan=json.loads((OUT/'cleanup_plan.json').read_text())
 # Keep the distinct X-audit report: it is not a duplicate hydrograph overview.
 plan=[r for r in plan if Path(r['path']).name!='15_panel_x_audit.html']
 allowed={ROOT/'RAPID/outputs/common_grid_production_integration_20261008'/s for s in ['base_smoke','base_replay','parent_child']}
 allowed|={ROOT/'RAPID/outputs/bounded_beta_calibration_20261009'/s for s in ['pre_completion_verification','verification_interrupted_by_concurrent_edit']}
 html_roots=[ROOT/'RAPID/outputs'/s for s in ['common_grid_validation_20261008','common_grid_height_calibration_20261008','bounded_beta_calibration_20261009']]
 files=[]
 for r in plan:
  p=Path(r['path']);assert not p.is_symlink()
  assert p in allowed or (p.is_file() and p.name.startswith('15_panel') and p.suffix=='.html' and any(p.is_relative_to(q) for q in html_roots))
  assert not subprocess.check_output(['git','ls-files','--',str(p.relative_to(ROOT))],cwd=ROOT).strip(),'Tracked files must be retained'
  contained=sorted(f for f in p.rglob('*') if f.is_file()) if p.is_dir() else [p]
  assert not any(f.is_symlink() for f in contained)
  assert sum(f.stat().st_size for f in contained)==r['bytes'],'Candidate changed after audit'
  files.extend(dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f),reason=r['reason']) for f in contained)
 protected={}
 for folder in [ROOT/'RAPID/outputs/bounded_beta_calibration_20261009',ROOT/'RAPID/outputs/common_grid_height_calibration_20261008/33847083c6ea']:
  for p in folder.glob('*/wave.parquet'):protected[str(p)]=sha(p)
 record=dict(status='prepared',canonical_overview=str(OUT/'15_network_overview.html'),files=files,
  total_bytes=sum(f['bytes'] for f in files),total_files=len(files),protected_wave_sha256=protected,
  preserved='Raw frozen inputs, selected parameters and waves, earlier calibration data, completed verification runs, X audit, example-11 diagnostic, source scripts, active geometry work.',
  regeneration='python RAPID/investigations/export_calibration_overview.py regenerates the canonical file. Historical plot exporters and source data retained; redundant historical HTML intentionally removed.')
 (OUT/'cleanup_manifest.json').write_text(json.dumps(record,indent=2))
 for r in plan:
  p=Path(r['path'])
  if p.is_dir():shutil.rmtree(p)
  else:p.unlink()
 for path,digest in protected.items():assert sha(Path(path))==digest
 assert sha(OUT/'15_network_overview.html')==manifest['html_sha256']
 record['status']='completed';(OUT/'cleanup_manifest.json').write_text(json.dumps(record,indent=2))
 (OUT/'cleanup_plan.json').unlink()
 print(f"Removed {record['total_files']} files, {record['total_bytes']/1024**2:.1f} MiB; protected selected waves unchanged.")
if __name__=='__main__':main()
