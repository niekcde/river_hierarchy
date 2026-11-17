"""Finite-pair Spearman associations for the broader descriptor audit."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import broad_descriptor_audit as b

def compute(frame,catalog,grid):
    rows=[]
    for (method,state),g in frame.groupby(['method','child_state']):
        xs=np.column_stack([g[k].map(dict(zip(levels,range(len(levels))))) if k=='internal_location' else g[k] for k,levels in grid.items()])
        cache={}
        for col in catalog.variable:
            a=g[col].to_numpy();valid=np.isfinite(a);v=a[valid];tol=1e-10*max(float(np.max(abs(v))) if len(v) else 0.,1.)
            key=np.packbits(valid).tobytes()
            if key not in cache:
                xx=rankdata(xs[valid],axis=0);xx-=xx.mean(axis=0) if len(xx) else 0
                cache[key]=(xx,np.sqrt(np.sum(xx**2,axis=0)))
            xx,norm=cache[key];rho=np.full(len(grid),np.nan)
            if len(v)>1 and np.ptp(v)>tol:
                yy=rankdata(np.round(v/tol)*tol);yy-=yy.mean();den=norm*np.linalg.norm(yy)
                np.divide(xx.T@yy,den,out=rho,where=den>0)
            rows.extend(dict(method=method,child_state=state,variable=col,factor=factor,spearman=rho[i],n_finite=int(valid.sum()),n_total=len(g)) for i,factor in enumerate(grid))
    return pd.DataFrame(rows)

def main():
    meta=json.loads((b.OUT/'provenance.json').read_text())
    for p,h in meta['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    m=json.loads((b.cp.OUTPUT/'manifest.json').read_text());f=pd.read_parquet(b.OUT/'features.parquet');cat=pd.read_csv(b.OUT/'candidate_catalog.csv')
    c=compute(f,cat,m['config']['grid']);c.to_csv(b.OUT/'control_correlations.csv',index=False)
    (b.OUT/'control_correlations_provenance.json').write_text(json.dumps(dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),b.OUT/'features.parquet',b.OUT/'provenance.json']},rows=len(c),statistic='Within-treatment/method Spearman, pairwise finite values; location ordered upstream/central/downstream; near ties rounded at relative scale 1e-10.'),indent=2))
    print('Control associations:',len(c))
if __name__=='__main__':main()
