"""Static geometric resistance with independently audited unit-flow energy bounds."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
from scopes import ROOT,OUT,terminals
from build import digest
from flow_resistance import solve_unit_flow
sys.path.insert(0,str(ROOT))
FLOW_FILE=Path(__file__).with_name('flow_resistance.py')
CONFIG=dict(tolerance=1e-8,max_iterations=250)
CACHE_VERSION=hashlib.sha256((digest(FLOW_FILE)+json.dumps(CONFIG,sort_keys=True)).encode()).hexdigest()[:16]

def solve_graph(graph,alpha):
    cache=OUT/'resistance_cache'/f'{graph}_a{alpha}_{CACHE_VERSION}.json'
    if cache.exists():return json.loads(cache.read_text())
    f=pd.read_parquet(OUT/'graphs'/f'{graph}.parquet');source,sink=terminals(f)
    edges=pd.DataFrame(dict(u=f.id_us_node,v=f.id_ds_node,K_e=f.link_length_m/f.width**(2+3*alpha)))
    start=time.monotonic();result=solve_unit_flow(edges,source,sink,**CONFIG)
    record=dict(graph=graph,alpha=alpha,K=float(result['K_eq']),lower_K=float(result['lower_K']),success=bool(result['success']),
        relative_energy_gap=float(result['relative_gap']),continuity=float(result['continuity']),iterations=int(result['iterations']),
        runtime_seconds=time.monotonic()-start,edges=len(f),solver_cache_version=CACHE_VERSION,
        graph_sha256=digest(OUT/'graphs'/f'{graph}.parquet'))
    if not np.isfinite(record['K']) or record['K']<=0:record['success']=False
    cache.write_text(json.dumps(record,indent=2)+'\n');return record

def main():
    (OUT/'resistance_cache').mkdir(exist_ok=True)
    geometry=pd.read_parquet(OUT/'geometry_responses.parquet')
    graphs=pd.read_csv(OUT/'domain_graphs.csv',dtype={'example':str})
    selected=graphs[graphs.parent_closed & graphs.child_closed]
    keys=sorted(set(selected.parent_graph)|set(selected.child_graph))
    solved=[]
    for key in keys:
        for alpha in [1.,.65]:solved.append(solve_graph(key,alpha))
        if len(solved)%500==0:print(len(solved),'/',2*len(keys),'static solves; failures',sum(not r['success'] for r in solved),flush=True)
    sol=pd.DataFrame(solved);sol.to_csv(OUT/'resistance_solver_audit.csv',index=False)
    ix=sol.set_index(['graph','alpha']);rows=[]
    for r in selected.itertuples():
        row=dict(example=r.example,state=r.state,scope=r.scope)
        for alpha,label in [(1.,'a1'),(.65,'a065')]:
            p=ix.loc[r.parent_graph,alpha];c=ix.loc[r.child_graph,alpha]
            ok=bool(p.success and c.success)
            row['K_ratio_'+label]=c.K/p.K if ok else np.nan
            row['K_log_'+label]=-np.log10(c.K/p.K) if ok else np.nan
            row['K_valid_'+label]=ok
        rows.append(row)
    resist=pd.DataFrame(rows);resist.to_csv(OUT/'resistance_by_scope.csv',index=False)
    wide=resist.set_index(['example','state','scope']).unstack('scope')
    wide.columns=[scope+'_'+metric for metric,scope in wide.columns]
    d=geometry.merge(wide.reset_index(),on=['example','state'],how='left',validate='one_to_one')
    for label in ['a1','a065']:
        for metric in ['K_ratio_','K_log_','K_valid_']:
            d['B_'+metric+label]=[r.get(r.B_scope+'_'+metric+label,np.nan) for _,r in d.iterrows()]
        d['B_weighted_K_log_'+label]=d.B_length_fraction*d.B_exposure*d['B_K_log_'+label]
    d.to_parquet(OUT/'linked_responses.parquet',index=False)
    (OUT/'resistance_manifest.json').write_text(json.dumps(dict(law='K_e proportional to L_e/W_e^(2+3alpha), wide Chezy with common scale cancelling in each ratio',
        alpha=[1.,.65],primary='alpha=1 legacy constant width/depth law; alpha=.65 explicit alternative',
        solver='Unit-flow convex energy minimization for same signed quadratic resistance; primal/dual bounds and mass balance checked',
        solver_config=CONFIG,solver_source_sha256=digest(FLOW_FILE),script_sha256=digest(__file__),
        geometry_sha256=digest(OUT/'geometry_responses.parquet'),graph_count=len(keys),solves=len(sol),failed_solves=int((~sol.success).sum()),
        note='Existing head-based solver had convergence failures on large compounds; this isolated numerical formulation is tested against analytic circuits and existing validated results. No archived K values, calibrated roughness, depth, or response variables used as predictors.'),indent=2)+'\n')
    print('Resistance complete:',len(sol),'solves;',int((~sol.success).sum()),'failures',flush=True)
if __name__=='__main__':main()
