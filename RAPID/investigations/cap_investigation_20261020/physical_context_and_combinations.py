"""Read-only diagnostics and geometry-selected augmentation inventory; no fits/routing."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'RAPID/outputs/uncapped_event_window_20261020'
DEST = OUT/'physical_context_and_combinations'
PAIR = ROOT/'RAPID/outputs/hydrograph_closed_unit_pairing_20261019'
EXT = ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019'
SOURCES = {}

def read(p):
    SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.suffix == '.json': return json.loads(p.read_text())
    if p.suffix == '.parquet': return pd.read_parquet(p)
    return pd.read_csv(p, dtype={'example':str})

def main():
    DEST.mkdir(exist_ok=True)
    blocks = read(PAIR/'eligible_blocks.json')
    plans = [read(ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json'),
             read(EXT/'routing_plan.json'),read(OUT/'routing_plan.json')]
    jobs = {}
    for plan in plans:
        for case in plan['cases']:
            jobs.setdefault(case['example'],{}).update({j['state_id']:j for j in case['jobs']})
    preds = read(OUT/'nine_input_tuning/predictions.parquet')
    preds = preds[(preds.variant=='same_grid') & (preds.cohort=='segmented')]
    phys=[]; sizes={}
    for ex, bb in blocks.items():
        f=read(Path(jobs[ex]['S000_base']['parameters']))
        ids=set(i for b in bb for i in b['parent_ids'])
        assert len(ids)==sum(len(b['parent_ids']) for b in bb)
        s=f[f.id_link.isin(ids)]
        w=s.reference_flow_fraction*s.link_length_m
        actual=float((w/s.rapid_celerity_mps).sum()/3600)
        # Diagnostic geometry-only counterfactual: fixed n=.03, h=3 m, no cap.
        # Not a fitted predictor, a new routing result, or a centroid prediction.
        standardized_c=(5/3)*3**(2/3)*np.sqrt(s.slope)/.03
        geom=float((w/standardized_c).sum()/3600)
        child=read(Path(jobs[ex]['all_closed_units_collapsed']['parameters']))
        lineage=read(EXT/f'variants/{ex}/all_closed_units_collapsed/lineage.csv')
        child_ids=lineage.loc[lineage.role.eq('collapsed_block'),'child_link_id'].unique()
        child=child[child.id_link.isin(child_ids)]
        cw=child.reference_flow_fraction*child.link_length_m
        child_time=float((cw/child.rapid_celerity_mps).sum()/3600)
        child_geom=float((cw/((5/3)*3**(2/3)*np.sqrt(child.slope)/.03)).sum()/3600)
        # Geometry-only joint width/slope quantity motivated by h proportional
        # to W**.65. The omitted depth scale and roughness remain diagnostics.
        gp=float((w/1000*(s.width/100)**(-13/30)/np.sqrt(s.slope)).sum())
        gc=float((cw/1000*(child.width/100)**(-13/30)/np.sqrt(child.slope)).sum())
        depth_factor=s.depth_m/s.width**.65
        r=preds[preds.example.eq(ex)].iloc[0]
        phys.append(dict(example=ex,edited_parent_km=w.sum()/1000,
            effective_celerity_mps=w.sum()/(actual*3600),parent_route_hours=actual,
            standardized_n_depth_route_hours=geom,hydraulic_to_standardized_time_ratio=actual/geom,
            n_manning=s.n_manning.iloc[0],route_time_change_h=child_time-actual,
            standardized_n_depth_time_change_h=child_geom-geom,
            geometry_width_slope_parent=gp,geometry_width_slope_delta=gc-gp,
            parent_time_per_geometry_unit=actual/gp,
            parent_depth_width_factor_cv=float(depth_factor.std(ddof=0)/depth_factor.mean()),
            route_delta_reconstruction_error_h=(actual/gp)*(gc-gp)-(child_time-actual),
            actual_centroid_h=r.centroid_shift_hours,
            predicted_centroid_h=r.predicted_hours,error_h=r.predicted_hours-r.centroid_shift_hours))
        for b in bb:
            z=f[f.id_link.isin(b['parent_ids'])]
            sizes[ex,b['state']]=float((z.reference_flow_fraction*z.link_length_m).sum()/1000)
    physical=pd.DataFrame(phys)
    np.testing.assert_allclose(physical.route_delta_reconstruction_error_h,0,atol=1e-10,rtol=0)
    physical.to_csv(DEST/'segmented_physical_context.csv',index=False)

    response=read(EXT/'responses.parquet')
    interaction_parameters=read(ROOT/'RAPID/outputs/example15_keep_ten_20261020/middle/parameters.parquet')
    interaction_regime=interaction_parameters[['celerity_max_mps','n_manning','diffusion_multiplier']].drop_duplicates().to_dict('records')
    prepared=read(PAIR/'prepared_jobs.json')
    interactions=[]
    for j in prepared:
        if j['kind']!='pair': continue
        for window in ['0.5%','1%','2%']:
            d=response[(response.example==j['example']) & (response.window==window)].set_index('state',verify_integrity=True)
            a,b=j['selected_states']; states=['control_'+a,'control_'+b,j['state']]
            z=d.loc[states]
            for c in ['baseline_cms','onset_utc','end_utc']:
                assert z[c].nunique()==1, (j['state'],c)
            shifts=z.centroid_shift_hours.to_numpy()*60
            interactions.append(dict(example=j['example'],state=j['state'],window=window,
                a_minutes=shifts[0],b_minutes=shifts[1],joint_minutes=shifts[2],
                interaction_minutes=shifts[2]-shifts[0]-shifts[1],
                regime='historical capped pilot; matched constructed single controls'))
    pd.DataFrame(interactions).to_csv(DEST/'existing_pair_interactions.csv',index=False)

    pairs=read(PAIR/'candidate_pairs.csv'); candidates=[]
    historical={(j['example'],tuple(j['selected_states'])) for j in prepared if j['kind']=='pair'}
    for ex,g in pairs.groupby('example'):
        arcs=list(g[['state_a','state_b']].itertuples(index=False,name=None))
        combos=set(arcs)
        # All contiguous chains, not arbitrary subsets or overlapping members.
        frontier=list(arcs)
        while frontier:
            chain=frontier.pop()
            for a,b in arcs:
                if chain[-1]==a and b not in chain:
                    new=(*chain,b)
                    if new not in combos:
                        combos.add(new);frontier.append(new)
        lookup={b['state']:b for b in blocks[ex]}
        for combo in sorted(combos):
            ids=[set(lookup[x]['parent_ids']) for x in combo]
            assert len(set.union(*ids))==sum(map(len,ids))
            candidates.append(dict(example=ex,states=' | '.join(combo),n_blocks=len(combo),
                edited_parent_km=sum(sizes[ex,x] for x in combo),
                block_types='+'.join(lookup[x]['kind'] for x in combo),
                historical_pair=(ex,combo) in historical,
                duplicates_segmented=set(combo)==set(lookup),
                status='adjacency inventory; combined geometry and hydraulic validation required'))
    c=pd.DataFrame(candidates); c.to_csv(DEST/'combination_inventory.csv',index=False)
    selected=[]
    for ex,g in c.groupby('example'):
        g=g[~g.duplicates_segmented]
        p=g[(g.n_blocks==2)&~g.historical_pair].sort_values(['edited_parent_km','states'])
        if len(p):
            for idx,reason in [(p.index[len(p)//2],'middle-sized new pair'),(p.index[-1],'largest new pair')]:
                row=p.loc[idx].to_dict();row['selection_reason']=reason;selected.append(row)
        t=g[g.n_blocks==3].sort_values(['edited_parent_km','states'])
        if len(t):
            row=t.iloc[-1].to_dict();row['selection_reason']='largest adjacent triple';selected.append(row)
    s=pd.DataFrame(selected).drop_duplicates(['example','states'])
    s.to_csv(DEST/'proposed_pilot.csv',index=False)
    scale=[]
    for ex,g in c.groupby('example'):
        g=g[(g.n_blocks>=4)&~g.duplicates_segmented].copy()
        total=sum(v for (e,_),v in sizes.items() if e==ex)
        for fraction in [.25,.5,.75]:
            if g.empty: continue
            g['distance_to_target']=abs(g.edited_parent_km-total*fraction)
            row=g.sort_values(['distance_to_target','n_blocks','states']).iloc[0].to_dict()
            row.update(target_fraction=fraction,actual_fraction=row['edited_parent_km']/total)
            scale.append(row)
    scale=pd.DataFrame(scale).drop_duplicates(['example','states'])
    assert not s.duplicates_segmented.any() and not scale.duplicates_segmented.any()
    assert not s.duplicated(['example','states']).any()
    assert not scale.duplicated(['example','states']).any()
    scale.to_csv(DEST/'proposed_scale_extension.csv',index=False)
    c.groupby(['example','n_blocks']).agg(candidates=('states','size'),max_edited_parent_km=('edited_parent_km','max')).to_csv(DEST/'coverage.csv')
    for path,h in SOURCES.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    manifest=dict(sources=SOURCES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        example15_interaction_settings=interaction_regime,
        candidate_count=len(c),proposed_count=len(s),scale_extension_count=len(scale),networks=s.example.nunique(),
        selection='Geometry-only: new median/largest pairs and largest triples; separate longer contiguous chains nearest 25/50/75% of edited parent size. Exclude duplicates of segmented full state. No response selection.',
        limitations='Historical pair interactions are not current uncapped labels. Candidate adjacency inherited from saved validated pair inventory; all new combinations still require full composition validation. No new routing or model fit.',
        production_modified=False)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(pd.DataFrame(phys).round(3).to_string(index=False))
    print(pd.DataFrame(interactions).query("window == '0.5%'").to_string(index=False))
    print('Inventory:',len(c),'Proposed:',len(s),'Networks:',s.example.nunique())

if __name__=='__main__':main()
