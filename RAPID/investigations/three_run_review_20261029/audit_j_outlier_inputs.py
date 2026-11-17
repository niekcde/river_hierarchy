"""Trace the selected outliers to saved head reconstruction and geometry inputs."""
import json
import numpy as np
import pandas as pd
from investigate_j_outlier_groups import DEST, ROOT, read_saved, edge_metrics, SOURCES
from incremental_storage import digest


def main():
    index_path=ROOT/'RAPID/outputs/compound_member_slopes_final_20261028/staged_slope_index.json'
    SOURCES[str(index_path)]=digest(index_path)
    index={r['case']:r for r in json.loads(index_path.read_text())['cases']}
    out=pd.read_csv(DEST/'outlier_diagnostics.csv',dtype={'example':str})
    summaries=[];tops=[];pairs=[]
    for r in out.itertuples():
        p=edge_metrics(read_saved(r.example,'S000_base'));c=edge_metrics(read_saved(r.example,r.state))
        assert p.n_manning.nunique()==c.n_manning.nunique()==1
        np.testing.assert_allclose(p.n_manning.iloc[0],c.n_manning.iloc[0])
        np.testing.assert_allclose(p.diffusion_multiplier.iloc[0],c.diffusion_multiplier.iloc[0])
        # Unique directed endpoint matches, with explicit tolerance for exported geometry.
        # Ambiguous parallel endpoint pairs are excluded, never matched by row order.
        pu=p[~p.duplicated(['u','v'],keep=False)]
        cu=c[~c.duplicated(['u','v'],keep=False)]
        m=pu.merge(cu,on=['u','v'],suffixes=('_parent','_child'),validate='one_to_one')
        stable=np.isclose(m.link_length_m_parent,m.link_length_m_child,rtol=1e-5,atol=.001)&np.isclose(m.width_parent,m.width_child,rtol=1e-5,atol=.001)
        m=m[stable].copy()
        m['slope_ratio']=m.slope_child/m.slope_parent
        m['allocation_ratio']=m.reference_flow_fraction_child/m.reference_flow_fraction_parent
        m['example']=r.example;m['state']=r.state;pairs.append(m)
        rec=dict(example=r.example,state=r.state,group=r.group,
            stable_geometry_directed_matches=len(m),slope_changed_gt1pct=int((abs(m.slope_ratio-1)>.01).sum()),
            allocation_changed_gt1pct=int((abs(m.allocation_ratio-1)>.01).sum()),
            ambiguous_parent_rows=int(p.duplicated(['u','v'],keep=False).sum()),
            ambiguous_child_rows=int(c.duplicated(['u','v'],keep=False).sum()))
        item=index.get(r.example+'/'+r.state)
        rec['staged_status']=item['status'] if item else 'not indexed: retained previous inputs'
        rec['frozen_floor']=np.nan;rec['floor_share_of_child_diffusion']=np.nan
        if item:
            sf=pd.read_parquet(item['slope_file']);audit=json.loads(open(item['audit_file']).read())
            for k in ['slope_file','audit_file','contract_file']:SOURCES[item[k]]=digest(item[k])
            rec['frozen_floor']=audit.get('solver',{}).get('frozen_floor',np.nan)
            # Match actual routed links to staged slopes by id, and verify correspondence.
            z=c.merge(sf[['id_link','slope','floor','role']],on='id_link',suffixes=('','_staged'),validate='one_to_one')
            np.testing.assert_allclose(z.slope,z.slope_staged,rtol=1e-7,atol=1e-12)
            atfloor=np.isclose(z.slope,z.floor,rtol=1e-5,atol=1e-12)&(z.floor>0)
            rec['floor_share_of_child_diffusion']=z.loc[atfloor,'diffusion_contribution_s2'].sum()/c.diffusion_contribution_s2.sum()
            rec['at_floor_reaches']=int(atfloor.sum())
        summaries.append(rec)
        top=c.nlargest(5,'diffusion_contribution_s2').copy()
        top['example']=r.example;top['state']=r.state;top['group']=r.group
        top['share_child_diffusion']=top.diffusion_contribution_s2/c.diffusion_contribution_s2.sum()
        top['prompt_cell_fraction']=np.where(top.rapid_x<0,-top.rapid_x/(1-top.rapid_x),np.nan)
        tops.append(top)
    pd.DataFrame(summaries).to_csv(DEST/'input_audit.csv',index=False)
    pd.concat(tops,ignore_index=True).to_csv(DEST/'dominant_child_reaches.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_parquet(DEST/'stable_geometry_pairs.parquet',index=False)
    (DEST/'input_audit_provenance.json').write_text(json.dumps(dict(source_sha256=SOURCES,script_sha256=digest(__file__),
        matching='Unique directed endpoints only; length and width rtol=1e-5, atol=0.001m. Reversed edges and ambiguous parallel endpoints excluded. Changed means >1% relative difference.',
        caveat='Original geometry_changes.csv uses strict signatures; new/removed there means unmatched signature, not necessarily physically added/deleted reaches.'),indent=2)+'\n')
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
