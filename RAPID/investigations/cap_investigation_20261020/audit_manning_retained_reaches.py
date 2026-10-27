"""Read-only audit of retained links in network-22 unit/compound collapses.

Island/control IDs can be regenerated and are deliberately excluded from this
ID-based audit. A match also requires identical endpoints, length and width.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from manning_beta1_network import OUT, ROOT, digest


def main():
    dest=OUT/'manning_retained_reach_audit_20261027'
    dest.mkdir(exist_ok=True)
    states=OUT/'manning_beta1_network22_20261027/states'
    sources={}
    def read(p):
        sources[str(p)]=digest(p)
        return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
    base=read(states/'S000_base/parameters.parquet')
    ownership=read(ROOT/'RAPID/assets/bounded_robust_slopes_20261002/22/floor_0.05/old/links.parquet')
    units=read(ROOT/'network_variants/output/22_indep__anchored/states/S000_base/hierarchy/unit_summary.csv')
    rows=[];detail=None
    for folder in sorted(states.iterdir()):
        if not folder.name.startswith(('S','CB')) or folder.name=='S000_base':continue
        child=read(folder/'parameters.parquet')
        m=base.merge(child,on='id_link',suffixes=('_base','_child'),validate='one_to_one')
        same=np.ones(len(m),dtype=bool)
        for k in ['u','v','link_length_m','width']:
            same &= np.isclose(m[k+'_base'],m[k+'_child'],rtol=1e-12,atol=1e-10)
        m=m[same].copy()
        for k in ['slope','reference_discharge_cms','depth_m','source_depth_m']:
            m[k+'_changed']=~np.isclose(m[k+'_base'],m[k+'_child'],rtol=1e-8,atol=1e-10)
        # At constant n and width the two drivers of depth changes are explicit.
        np.testing.assert_allclose(m.depth_m_child/m.depth_m_base,
            (m.reference_discharge_cms_child/m.reference_discharge_cms_base)**.6
            *(m.slope_child/m.slope_base)**(-.3),rtol=1e-10)
        rows.append(dict(state=folder.name,retained_same_geometry=len(m),
            **{k:int(m[k].sum()) for k in ['slope_changed','reference_discharge_cms_changed','depth_m_changed','source_depth_m_changed']},
            max_depth_change_m=float((m.depth_m_child-m.depth_m_base).abs().max())))
        if folder.name=='S296_unit_30':
            m=m.merge(ownership[['id_link','ownership_kind','ownership_domain_id']],on='id_link',validate='one_to_one')
            nodes=set(map(int,units.loc[units.unit_id.eq(30),'unit_node_ids'].iloc[0].split(',')))
            m['inside_selected_unit']=m.u_base.isin(nodes)&m.v_base.isin(nodes)
            m.to_csv(dest/'unit30_retained_reaches.csv',index=False)
            changed=m[m.depth_m_changed]
            assert len(m)==949 and len(changed)==470
            assert not changed.inside_selected_unit.any()
            assert changed.ownership_kind.eq('compound_bubble').all() and changed.ownership_domain_id.eq(9).all()
            assert not m.source_depth_m_changed.any()
            exterior=m[~(m.ownership_kind.eq('compound_bubble')&m.ownership_domain_id.eq(9))]
            assert not exterior[['slope_changed','reference_discharge_cms_changed','depth_m_changed']].any().any()
            detail=dict(state=folder.name,retained_same_geometry=len(m),
                changed_depth_outside_selected_unit=len(changed),changed_slope=int(m.slope_changed.sum()),
                changed_reference_discharge=int(m.reference_discharge_cms_changed.sum()),
                max_depth_change_m=float((m.depth_m_child-m.depth_m_base).abs().max()),
                containing_compound=9,retained_outside_compound=len(exterior),
                outside_compound_unchanged=True,BASED_retained_depths_unchanged=True)
    frame=pd.DataFrame(rows);frame.to_csv(dest/'state_summary.csv',index=False)
    manifest_path=ROOT/'RAPID/assets/bounded_robust_slopes_20261002/child_head_reconstruction_20261003/22/contexts/434bbf68974ad00f2984/hydraulic_context_manifest.json'
    sources[str(manifest_path)]=digest(manifest_path)
    head_audit=json.loads(manifest_path.read_text())['head_audit']
    for p in [Path(__file__),ROOT/'network_variants/shared_head_reconstruction.py',ROOT/'RAPID/src/rapid_tools/cunge.py']:
        sources[str(p)]=digest(p)
    result=dict(complete=True,scope='Network22 saved beta=1 Manning unit and compound cases; stable-ID matches additionally checked for identical endpoints, length and width. Island/control states excluded.',
        cases=len(frame),cases_with_retained_depth_changes=int(frame.depth_m_changed.gt(0).sum()),
        unit30=detail,unit30_saved_head_audit=head_audit,
        finding='The containing compound is re-solved for shared heads; retained slopes change outside the selected unit. Width-based reference-flow allocation also changes inside the compound. Manning depth inherits both changes. These slopes and allocations already exist in BASED inputs; retained BASED depths are unchanged because their width is unchanged.',
        implication='Current responses are not strictly local hydraulic interventions. Decide the permitted hydraulic footprint before the all-network rerun. Freezing depth alone would leave slope and flow-allocation changes.',
        source_hashes=sources,production_modified=False,rerouted_states=0)
    for p,h in sources.items():assert digest(p)==h
    (dest/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','unit30_saved_head_audit']},indent=2))


if __name__=='__main__':main()
