"""Wait for numerical jobs, regenerate final artifacts, and exercise offline plots."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/broad_15_100m'
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

def ready():
    p=OUT/'calibrated_grid_checks.json'
    return (OUT/'stages_complete.json').exists() and p.exists() and '37' in {r['example'] for r in json.loads(p.read_text())}

def browser_check(name):
    source=OUT/name
    stem=source.stem
    temp=Path('/tmp')/f'{stem}_final_qa.html'
    javascript=r"""
<script>
window.qaErrors=[];
window.addEventListener('error',e=>qaErrors.push(e.message));
window.addEventListener('unhandledrejection',e=>qaErrors.push(String(e.reason)));
window.addEventListener('load',()=>setTimeout(async()=>{
 const result={cases:[]};
 try {
  const gd=document.querySelector('.plotly-graph-div');
  const buttons=gd.layout.updatemenus?.[0]?.buttons||[];
  for(const item of buttons){
   document.querySelector('.updatemenu-header').dispatchEvent(new MouseEvent('click',{bubbles:true}));
   await new Promise(r=>setTimeout(r,80));
   const button=[...document.querySelectorAll('.updatemenu-dropdown-button')].find(x=>x.textContent.trim()===item.label);
   if(!button)throw Error('Missing menu item '+item.label);
   button.dispatchEvent(new MouseEvent('click',{bubbles:true}));
   await new Promise(r=>setTimeout(r,200));
   const title=gd._fullLayout.title.text;
   const expected=item.args[0].visible.filter(Boolean).length;
   const visible=gd.data.filter(x=>x.visible!==false&&x.visible!=='legendonly').length;
   const expectedTitle=item.args[1].title.text;
   result.cases.push({label:item.label,title,expected,visible,passed:title===expectedTitle&&visible===expected});
  }
  result.renderedPaths=gd.querySelectorAll('.scatterlayer path').length;
  result.passed=result.cases.every(x=>x.passed)&&result.renderedPaths>0&&qaErrors.length===0;
 }catch(e){result.error=String(e);result.passed=false;}
 result.errors=qaErrors;
 const p=document.createElement('pre');p.id='plot-qa-result';p.textContent=JSON.stringify(result);document.body.appendChild(p);
},1000));
</script>
"""
    temp.write_text(source.read_text().replace('</body>',javascript+'</body>'))
    command=[CHROME,'--headless','--disable-gpu','--disable-background-networking','--no-first-run','--disable-sync','--disable-extensions','--user-data-dir=/tmp/lisflood-final-review-profile','--dump-dom',f'--screenshot={OUT/(stem+"_final_check.png")}','--window-size=1440,1500','--virtual-time-budget=18000',temp.as_uri()]
    run=subprocess.run(command,capture_output=True,text=True,timeout=60)
    match=re.search(r'<pre id="plot-qa-result">(.*?)</pre>',run.stdout,re.S)
    if run.returncode or not match:
        raise RuntimeError(f'{name}: browser QA failed: {run.stderr[-1500:]}')
    result=json.loads(html.unescape(match[1]));result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT/(stem+'_final_qa.json')).write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError(f'{name}: {result}')
    print('BROWSER PASSED',name,len(result['cases']),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    start=time.monotonic()
    while not ready():
        if not args.wait:raise RuntimeError('Numerical stages are not complete')
        if time.monotonic()-start>86400:raise RuntimeError('Numerical stage wait exceeded one day')
        time.sleep(30)
    for name in ['summarize_broad.py','plot_broad_grid.py','forecast_broad_runtime.py','report_broad.py']:
        subprocess.run([sys.executable,str(ROOT/name)],check=True)
    checks={name:browser_check(name) for name in ['hydrographs_plotly.html','grid_sensitivity_plotly.html','collapse_comparison_plotly.html']}
    source=OUT/'source_snapshot'
    for p in source.glob('*.py'):
        if (ROOT/p.name).exists():shutil.copy2(ROOT/p.name,p)
    shutil.copy2(Path(__file__),source/Path(__file__).name)
    p=OUT/'protocol.json';protocol=json.loads(p.read_text());protocol['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')};protocol['post_calibration_mesh_examples']=['02','03','15','37','56'];p.write_text(json.dumps(protocol,indent=2)+'\n')
    result=dict(complete=True,browser_checks_passed=True,summary=json.loads((OUT/'summary_status.json').read_text()),browser_source_hashes={name:r['source_sha256'] for name,r in checks.items()})
    (OUT/'review_complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FINAL REVIEW COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
