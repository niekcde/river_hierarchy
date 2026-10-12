"""Guarded switch of the unstarted master batch to calibrated Cunge only."""
from pathlib import Path
import json, hashlib, sys
ROOT=Path(__file__).resolve().parents[3]
STAGE=Path(__file__).parent
pointer=ROOT/'RAPID/config/production_run.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def encode(d):return json.dumps(d,indent=2)+'\n'
if sys.argv[1]=='stage':
 ptr=json.loads(pointer.read_text());manifest=Path(ptr['manifest']);m=json.loads(manifest.read_text());master=Path(m['master_root']);run=manifest.parent
 assert sha(manifest)==ptr['sha256'] and m['methods']==['cunge','single_x']
 for key in ['prep','results']:
  assert not any(p.is_file() for p in (master/m['outputs'][key]).rglob('*')), 'Batch already started'
 (STAGE/'previous_run_manifest.json').write_bytes(manifest.read_bytes())
 m['methods']=['cunge'];new=encode(m);newsha=hashlib.sha256(new.encode()).hexdigest()
 ptr['sha256']=newsha
 current=master/'rapid_runs/current.json';cur=json.loads(current.read_text());cur['sha256']=newsha
 info=run/'software_integration.json';e=json.loads(info.read_text());e['run_manifest_sha256']=newsha;e['selected_production_methods']=['cunge'];e['method_selection_note']='Historical smoke/resume evidence covers both supported methods; production now selects Cunge only.'
 counts=run/'reports/inventory_overview.json';c=json.loads(counts.read_text());c.update(method_jobs=2187,methods=['cunge'],base_jobs=15)
 readme=run/'README.txt';s=readme.read_text().replace('shared native clocks across the two methods. Base n/beta are inherited; no children\nare fitted. Cunge retains its selected subdivisions; the single-X comparator\nretains its own calibrated grid policy.', 'shared native clocks across compared states. Base n/beta are inherited; no children\nare fitted. Only calibrated Cunge is selected for production, retaining its selected\nsubdivisions.').replace('4,374 method jobs for Cunge plus single X.', '2,187 Cunge jobs, including 15 shared base jobs.').replace('Validation: 15 targeted tests;', 'Historical two-method integration validation: 15 targeted tests;')
 report=run/'reports/inventory_overview.html';h=report.read_text().replace('Two methods give 4,374 routing jobs.', 'Cunge only gives 2,187 routing jobs, including 15 shared base jobs.')
 files={manifest:new,current:encode(cur),pointer:encode(ptr),info:encode(e),counts:encode(c),readme:s,report:h}
 plan=[]
 for i,(p,content) in enumerate(files.items()):
  staged=STAGE/f'{i}_{p.name}';staged.write_text(content);plan.append(dict(path=str(p),staged=str(staged),before=sha(p),after=sha(staged)))
 (STAGE/'plan.json').write_text(encode(plan))
 print('Staged Cunge-only manifest, pointers and documentation; 2,187 jobs.')
else:
 plan=json.loads((STAGE/'plan.json').read_text())
 for r in plan:assert sha(Path(r['path']))==r['before'] and sha(Path(r['staged']))==r['after']
 for r in plan:
  p=Path(r['path']);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(Path(r['staged']).read_bytes());tmp.replace(p);assert sha(p)==r['after']
 (STAGE/'receipt.json').write_text(encode(dict(complete=True,files=plan)))
 print('Published Cunge-only configuration.')
