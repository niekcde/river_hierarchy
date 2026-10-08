"""Read-only source audit and fixed, exploratory CV; never routes or calibrates.

Run with the river-hierarchy-rivgraph interpreter. Every run creates a new output
folder. Grouping uses base unit edge-overlap components, alias memberships,
and overlapping matched contexts. Incomplete edit lineage is audited separately.
"""
from pathlib import Path
import sys, json, hashlib, itertools, ast
from datetime import datetime
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'RAPID/src'))
from rapid_tools.path_geometry import path_geometry
SOURCE = ROOT/'RAPID/outputs/cunge_fixed_comparison_plus37_20261006'
PREVIOUS = ROOT/'RAPID/outputs/cunge_fixed_extra_trees_plus37_20261006'
OUT = ROOT/'RAPID/outputs'/('prediction_audit_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
HASHES = {}

def track(path):
    p = Path(path)
    HASHES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def read(path):
    p = track(path)
    return pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)

def ids(value):
    if pd.isna(value) or not str(value).strip(): return set()
    return {int(float(x)) for x in str(value).replace('[','').replace(']','').split(',') if x.strip()}

def metrics(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    mae = np.average(abs(y-p), weights=w); zero = np.average(abs(y), weights=w)
    return dict(r2=r2_score(y,p,sample_weight=w) if len(y)>1 and np.ptp(y)>0 else np.nan,
                mae=float(mae), zero_mae=float(zero), skill_vs_zero=float(1-mae/zero) if zero else np.nan)

def graph(frame):
    g=nx.MultiDiGraph()
    for r in frame.itertuples():
        g.add_edge(int(r.id_us_node),int(r.id_ds_node),key=int(r.id_link),length=float(r.link_length_m),width=float(r.width))
    return g

def topology(frame):
    """Counts preserve parallel channels; degree-2 edge subdivision invariant."""
    g=graph(frame); total=frame.link_length_m.sum(); bif=sum(g.out_degree(n)>1 for n in g)
    result=dict(edges=g.number_of_edges(),nodes=len(g),bifurcations=bif,
                bifurcations_per_km=bif/(total/1000) if total else np.nan,
                independent_cycles=g.number_of_edges()-len(g)+nx.number_weakly_connected_components(g),
                length_m=total)
    sources=[n for n in g if g.in_degree(n)==0];sinks=[n for n in g if g.out_degree(n)==0]
    result.update(log1p_paths=np.nan,width_markov_entropy=np.nan,width_markov_length_cv=np.nan)
    if len(sources)!=1 or len(sinks)!=1 or not nx.is_directed_acyclic_graph(g): return result
    source,sink=sources[0],sinks[0]; order=list(nx.topological_sort(g));count={sink:1};mu={sink:0.};var={sink:0.};ent={sink:0.}
    for n in reversed(order[:-1]):
        e=list(g.out_edges(n,keys=True,data=True)); widths=np.array([x[3]['width'] for x in e]);p=widths/widths.sum()
        count[n]=sum(count[v] for _,v,k,d in e)
        mu[n]=sum(q*(d['length']+mu[v]) for q,(_,v,k,d) in zip(p,e))
        var[n]=sum(q*(var[v]+(d['length']+mu[v]-mu[n])**2) for q,(_,v,k,d) in zip(p,e))
        ent[n]=-sum(p*np.log(p))+sum(q*ent[v] for q,(_,v,k,d) in zip(p,e))
    result.update(log1p_paths=float(np.log1p(float(count[source]))),width_markov_entropy=ent[source],width_markov_length_cv=np.sqrt(var[source])/mu[source])
    return result

def synthetic_checks():
    rows=[]
    for stages in [1,2,8,32]:
        records=[]
        for j in range(stages):
            for branch in range(2): records.append(dict(id_link=2*j+branch,id_us_node=j,id_ds_node=j+1,link_length_m=100.,width=10.,head_us=stages-j,head_ds=stages-j-1))
        f=pd.DataFrame(records);a,_,_=path_geometry(f,list(f.id_link));t=topology(f)
        np.testing.assert_allclose([a['equivalent_length'],a['elongation'],a['dominant_width_fraction']], [100*stages,10*stages/2**stages,1/2**stages],rtol=1e-8)
        split=[]
        for r in f.itertuples():
            mid=1000+r.id_link
            for k,u,v,hu,hd in [(2*r.id_link,r.id_us_node,mid,r.head_us,(r.head_us+r.head_ds)/2),(2*r.id_link+1,mid,r.id_ds_node,(r.head_us+r.head_ds)/2,r.head_ds)]:
                split.append(dict(id_link=k,id_us_node=u,id_ds_node=v,link_length_m=50.,width=10.,head_us=hu,head_ds=hd))
        sf=pd.DataFrame(split);b,_,_=path_geometry(sf,list(sf.id_link));st=topology(sf)
        for col in ['equivalent_length','elongation','dominant_width_fraction','path_length_cv','association','paths']:np.testing.assert_allclose(a[col],b[col],rtol=1e-8,atol=1e-10)
        for col in ['bifurcations','bifurcations_per_km','independent_cycles','log1p_paths','width_markov_entropy','width_markov_length_cv']:np.testing.assert_allclose(t[col],st[col],atol=1e-10)
        rows.append(dict(serial_diamonds=stages,**a,**t,subdivision_pass=True))
    pd.DataFrame(rows).to_csv(OUT/'synthetic_geometry_checks.csv',index=False)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    track(__file__)
    for rel in ['RAPID/src/rapid_tools/path_geometry.py','RAPID/src/rapid_tools/refreshed_analysis.py','RAPID/src/rapid_tools/plotting.py','hierarchy_level_definition/unit_detection/bifurcation_confluence_units.py','RAPID/investigations/full_population_lag_search.py']:
        track(ROOT/rel)
    raw=read(SOURCE/'paired_results.parquet');d=raw[raw.method.eq('cunge')].copy().reset_index(drop=True)
    m=read(PREVIOUS/'matched_features_responses.parquet');pred=read(PREVIOUS/'held_out_predictions.parquet')
    meta=json.loads(track(PREVIOUS/'provenance.json').read_text());cols=meta['geometry_features']
    assert len(d)==552 and not d.duplicated(['example_id','fingerprint']).any()
    assert not d.state_id.eq('island_37_151').any()
    # Recompute the saved prediction scores independently (no refitting).
    verification=[]
    for (routing,response,features),g in pred.groupby(['routing','response','features']):
        for scope,z in [('pooled',g),('37',g[g.example_id.eq('37')]),('pooled_without_CB001',g[~(g.example_id.eq('37')&g.state_id.eq('CB001_bubble_3'))]),('37_without_CB001',g[g.example_id.eq('37')&~g.state_id.eq('CB001_bubble_3')])]:
            for units,y,p in [('target',z.observed,z.predicted),('hours',z.observed_shift_h,z.predicted_shift_h)]:verification.append(dict(routing=routing,response=response,features=features,scope=scope,units=units,n=len(z),**metrics(y,p)))
    v=pd.DataFrame(verification);v.to_csv(OUT/'verified_existing_scores.csv',index=False)
    for response,units,expected in [('shift_h','target',.042102),('relative_shift_pct','target',.069668),('relative_shift_pct','hours',.126290)]:
        value=v[(v.routing=='cunge')&(v.response==response)&(v.features=='geometry_15')&(v.scope=='pooled')&(v.units==units)].r2.iloc[0];assert abs(value-expected)<1e-6
    range_rows=[]
    for (method),z in m.groupby('method'):
        train=z[~z.example_id.eq('37')];test=z[z.example_id.eq('37')]
        for c in ['shift_h','relative_shift_pct']+cols:
            range_rows.append(dict(method=method,variable=c,train_min=train[c].min(),train_max=train[c].max(),test_min=test[c].min(),test_max=test[c].max(),outside=int(((test[c]<train[c].min())|(test[c]>train[c].max())).sum())))
    pd.DataFrame(range_rows).to_csv(OUT/'range_audit.csv',index=False)
    graph_groups=nx.Graph();graph_groups.add_nodes_from(d.index)
    grouping=[];footprints={};domains={};aliases=[];unit_checks=[];overlaps=[];topos=[];nested=[];frame_change=[]
    for ex,z in d.groupby('example_id'):
        root=ROOT/f'network_variants/output/{ex}_indep__anchored/states/S000_base'
        units=read(root/'hierarchy/unit_summary.csv').set_index('unit_id');membership=read(root/'variant/summary/unit_link_membership.csv')
        unit_edges={int(u):set(s.id_link.astype(int)) for u,s in membership.groupby('unit_id')}
        ug=nx.Graph();ug.add_nodes_from(units.index)
        for a,b in itertools.combinations(units.index,2):
            if unit_edges[a]&unit_edges[b]:ug.add_edge(a,b)
        components=list(nx.connected_components(ug))
        assert len(components)==units.compound_bubble_id.nunique()
        for cc in components:assert units.loc[list(cc),'compound_bubble_id'].nunique()==1
        for uid,u in units.iterrows():
            for child in ast.literal_eval(u.children):
                assert units.loc[child,'compound_bubble_id']==u.compound_bubble_id
                nested.append(dict(example=ex,parent_unit=uid,child_unit=child,same_group=True,child_edges_subset=unit_edges[child]<=unit_edges[uid]))
        for bubble,uu in units.groupby('compound_bubble_id'):
            unit_checks.append(dict(example=ex,bubble=int(bubble),units=len(uu),shared_edge_components_verified=True))
        refresh=Path(z.iloc[0].refreshed_run_dir).parents[1];inv=read(refresh/'inventory.csv')
        ready=inv[inv.status.eq('ready')].drop_duplicates('fingerprint').set_index('fingerprint')
        base=read(ready[ready.state.eq('S000_base')].iloc[0].frame);base_ids=set(base.id_link);bi=base.set_index('id_link');total=base.link_length_m.sum()
        registries={mode:read(ROOT/f'network_variants/output/{ex}_{family}__anchored'/filename).set_index(index) for mode,family,filename,index in [('indep','indep','state_registry.csv','state_id'),('compound','compound_bubbles','state_registry.csv','state_id'),('island','island','island_analysis_manifest_strict.csv','island_id')]}
        cache={}
        for i,r in z.iterrows():
            selected=set();bubbles=set()
            for a in json.loads(r.selection_aliases):
                reg=registries[a['mode']].loc[a['state_id']]
                uids={int(reg.source_unit_id)} if a['mode']=='island' else ids(reg.selected_unit_ids)
                assert uids and uids<=set(units.index)
                bs=set(units.loc[list(uids),'compound_bubble_id'].astype(int));bubbles|=bs;selected|=uids
                aliases.append(dict(example=ex,fingerprint=r.fingerprint,alias=a['state_id'],mode=a['mode'],unit_ids=','.join(map(str,sorted(uids))),bubble_ids=','.join(map(str,sorted(bs)))))
            assert int(r.compound_bubble_id) in bubbles
            support=set().union(*(unit_edges[u] for u in selected))
            child=read(ready.loc[r.fingerprint,'frame']);ci=child.set_index('id_link');common=bi.index.intersection(ci.index)
            geom_cols=['id_us_node','id_ds_node','link_length_m','width']
            naive_changed=set(common[~np.isclose(bi.loc[common,geom_cols],ci.loc[common,geom_cols],rtol=1e-10,atol=1e-9).all(axis=1)])
            variant=Path(ready.loc[r.fingerprint,'manifest']).parent.parent/'variant'
            lineage_paths=list(variant.glob('*/link_lineage.csv'))
            assert len(lineage_paths)==1, (r.state_id,lineage_paths)
            lineage=read(lineage_paths[0]);unmapped_children=set(child.id_link)-set(lineage.id_link)
            changed=set();represented=set();renumbered=0;unchanged_pairs=[]
            for link in lineage.itertuples():
                parents=ids(link.matched_parent_link_ids);assert parents<=base_ids
                represented|=parents
                if link.lineage_type!='unchanged_1to1':changed|=parents
                else:
                    assert len(parents)==1
                    pid=next(iter(parents));renumbered+=int(pid!=link.id_link)
                    if link.id_link not in ci.index:continue
                    unchanged_pairs.append((pid,link.id_link))
            if unchanged_pairs:
                parent_ids,child_ids=map(list,zip(*unchanged_pairs))
                differs=~np.isclose(bi.loc[parent_ids,['link_length_m','width']].to_numpy(),ci.loc[child_ids,['link_length_m','width']].to_numpy(),rtol=1e-8,atol=1e-7).all(axis=1)
                changed.update(np.asarray(parent_ids)[differs])
            removed=base_ids-represented
            # Some routed frames contain post-lineage anchor splits. Therefore these
            # mappings are an audit, not a complete measured modification footprint.
            # Group on stable base selection supports and complete matched contexts.
            footprint=support;footprints[i]=footprint;domains[i]=ids(r.matched_base_link_ids)
            frame_change.append(dict(example=ex,state_id=r.state_id,selected_support_edges=len(support),unrepresented_base_edges=len(removed),mapped_changed_base_edges=len(changed),mapped_changed_outside_selected=len(changed-support),unmapped_child_edges=len(unmapped_children),renumbered_unchanged_edges=renumbered,naive_same_id_changed_edges=len(naive_changed),selected_length_fraction=bi.loc[list(support),'link_length_m'].sum()/total,mapped_changed_base_length_fraction=bi.loc[list(changed),'link_length_m'].sum()/total,matched_length_fraction=r.length_fraction))
            grouping.append(dict(row=i,example_id=ex,fingerprint=r.fingerprint,state_id=r.state_id,bubble_ids=','.join(map(str,sorted(bubbles))),selected_unit_ids=','.join(map(str,sorted(selected))),footprint_ids=','.join(map(str,sorted(footprint))),unit_type=r.target_type))
            for role,s in [('selected_support',support),('matched_context',domains[i])]:
                key=tuple(sorted(s))
                if key not in cache:cache[key]=topology(base[base.id_link.isin(s)])
                topos.append(dict(example=ex,state_id=r.state_id,role=role,**cache[key]))
        local={r['row']:r for r in grouping if r['example_id']==ex}
        for a,b in itertools.combinations(z.index,2):
            same_bubble=bool(ids(local[a]['bubble_ids'])&ids(local[b]['bubble_ids']));physical=bool(footprints[a]&footprints[b]);context=bool(domains[a]&domains[b])
            if same_bubble or physical or context:graph_groups.add_edge(a,b)
            if context or physical:
                overlaps.append(dict(example=ex,row_a=a,row_b=b,same_stored_bubble=same_bubble,physical_overlap=physical,matched_domain_overlap=context))
        print('Audited',ex,len(z),'comparisons',flush=True)
    group_map={i:f'{d.loc[min(cc),"example_id"]}:G{min(cc):04d}' for cc in nx.connected_components(graph_groups) for i in cc}
    groups=pd.DataFrame(grouping).sort_values('row');groups['group']=groups.row.map(group_map)
    d['group']=d.index.map(group_map)
    oa=pd.DataFrame(overlaps);oa['same_final_group']=[group_map[a]==group_map[b] for a,b in zip(oa.row_a,oa.row_b)]
    # Conservative overlap audit gate: do not implement CV if a matched context crosses groups.
    assert oa.loc[oa.physical_overlap,'same_final_group'].all()
    for name,table in [('groups',groups),('alias_audit',pd.DataFrame(aliases)),('unit_membership_audit',pd.DataFrame(unit_checks)),('nested_membership_audit',pd.DataFrame(nested)),('overlap_audit',oa),('modification_footprint_audit',pd.DataFrame(frame_change)),('topology_candidates',pd.DataFrame(topos))]:table.to_csv(OUT/f'{name}.csv',index=False)
    summary=d.groupby(['example_id','group']).agg(pairs=('state_id','size'),types=('target_type',lambda x:','.join(sorted(set(x)))),min_shift_h=('shift_h','min'),max_shift_h=('shift_h','max')).reset_index();summary.to_csv(OUT/'group_sizes.csv',index=False)
    # Pairwise correlations and linear rank are descriptive only, not feature selection.
    x=m[m.method.eq('cunge')].merge(groups[['example_id','fingerprint','group']],on=['example_id','fingerprint'],validate='one_to_one')
    correlations=[];ranks=[]
    for scope,z in [('pooled',x)]+list(x.groupby('example_id')):
        for method in ['pearson','spearman']:
            cor=z[cols].corr(method=method)
            for a,b in itertools.combinations(cols,2):correlations.append(dict(scope=scope,method=method,feature_a=a,feature_b=b,correlation=cor.loc[a,b]))
        std=z[cols].std(ddof=0);active=std[std>1e-12].index;matrix=(z[active]-z[active].mean())/std[active];s=np.linalg.svd(matrix,compute_uv=False)
        ranks.append(dict(scope=scope,rows=len(z),nonconstant_features=len(active),rank=int(np.linalg.matrix_rank(matrix)),condition_number=float(s[0]/s[-1]) if s[-1]>0 else np.inf,constant_features=','.join(std[std<=1e-12].index)))
    pd.DataFrame(correlations).to_csv(OUT/'feature_correlations.csv',index=False);pd.DataFrame(ranks).to_csv(OUT/'feature_rank.csv',index=False)
    identities=[]
    for name,a,b in [('elongation_width_length',np.log10(d.parent_elongation),np.log10(d.parent_equivalent_length)-np.log10(d.parent_equivalent_length/d.parent_elongation)),('change_elongation',d.log_elongation_ratio,np.log(d.child_elongation/d.parent_elongation)),('exposure_length',d.length_fraction,d.matched_len_frac),('omega_product',d.omega,d.resistance_log*d.length_fraction*d.width_fraction)]:identities.append(dict(identity=name,max_absolute_error=float(np.max(abs(a-b)))))
    pd.DataFrame(identities).to_csv(OUT/'feature_identities.csv',index=False)
    synthetic_checks()
    if not oa.same_final_group.all():raise ValueError('Cross-group matched-domain overlap requires review before CV')
    evaluate(m,groups,cols,d)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in HASHES.items()),'Source changed during audit'
    (OUT/'provenance.json').write_text(json.dumps(dict(source_hashes=HASHES,source_unchanged=True,python=sys.version,pairs=len(d),groups=len(summary),scope='No routing, calibration or production writes. Fixed exploratory models; no tuning or feature selection. Current event and calibrated reference hydraulics shared within network.'),indent=2))
    print('COMPLETE',OUT,flush=True)


def evaluate(m,groups,cols,d):
    # Freeze structure-only splits: response bins are reported, not optimized.
    identity=groups[['example_id','fingerprint','group']]
    base=m[m.method.eq('cunge')].merge(identity,on=['example_id','fingerprint'],validate='one_to_one').sort_values(['example_id','fingerprint']).reset_index(drop=True)
    splits=list(GroupKFold(n_splits=5).split(base,groups=base.group));fold_by_key={}
    audits=[]
    for fold,(train,test) in enumerate(splits):
        tr,te=base.iloc[train],base.iloc[test];assert set(tr.group).isdisjoint(te.group)
        for r in te.itertuples():fold_by_key[(r.example_id,r.fingerprint)]=fold
        audits.append(dict(fold=fold,train_rows=len(tr),test_rows=len(te),train_groups=tr.group.nunique(),test_groups=te.group.nunique(),test_networks=','.join(sorted(te.example_id.unique())),test_37=int(te.example_id.eq('37').sum()),train_response_min=tr.relative_shift_pct.min(),train_response_max=tr.relative_shift_pct.max(),test_response_min=te.relative_shift_pct.min(),test_response_max=te.relative_shift_pct.max()))
    pd.DataFrame(audits).to_csv(OUT/'grouped_fold_audit.csv',index=False)
    plan=base[['example_id','fingerprint','state_id','group','relative_shift_pct']].copy();plan['fold']=[fold_by_key[(r.example_id,r.fingerprint)] for r in plan.itertuples()]
    plan=plan.merge(d[['example_id','fingerprint','target_type']],on=['example_id','fingerprint'],validate='one_to_one')
    plan['signed_bin']=pd.cut(plan.relative_shift_pct,[-np.inf,-5,-1,-.1,.1,1,5,np.inf],right=False).astype(str)
    plan.to_csv(OUT/'fold_assignment.csv',index=False)
    plan.groupby(['fold','target_type','signed_bin']).agg(pairs=('group','size'),groups=('group','nunique')).reset_index().to_csv(OUT/'stratification_diagnostic.csv',index=False)
    families={'resistance':['resistance_log'],'exposure':['length_fraction','width_fraction'],'parent_shape':['log10_parent_equivalent_length','log10_parent_elongation','log10_parent_dominant_width_fraction','parent_path_length_cv','signed_length_width_association'],'reference_slope':['log10_domain_reference_slope'],'geometry_change':['log_equivalent_length_ratio','log_elongation_ratio','change_path_length_cv','change_width_dominance'],'position':['log10_1p_upstream_shortest_length_m','log10_1p_downstream_shortest_length_m']}
    assert set(sum(families.values(),[]))==set(cols)
    variants={'geometry_15':cols,'geometry_15_plus_n':cols+['n_supplied'],**{f'without_{name}':[c for c in cols if c not in cs] for name,cs in families.items()}}
    (OUT/'evaluation_plan.json').write_text(json.dumps(dict(estimator=dict(n_estimators=96,min_samples_leaf=5,max_features=1.,random_state=20261005,n_jobs=1),families=families,variants=variants,selection='None; exploratory fixed ablations, no winner selected.',grouping='Network-qualified shared-edge unit components, unioned through all aliases, base selection-support overlap, and matched-context overlap. Partial edit lineage is audited, not used as a complete footprint.',outer='Five GroupKFold folds pooled across networks, plus separate leave-one-network-out ablations.',stratification='Structure-only assignment frozen; signed Cunge percentage bins and target type audited afterward. Cannot spread the giant group. No response-driven fold search.',inner_future='Any selection/tuning must use grouped inner folds of outer training only, optimize equal-group MAE, then refit once and evaluate untouched outer test.'),indent=2))
    predictions=[]
    for method,z in m.groupby('method'):
        z=z.merge(identity,on=['example_id','fingerprint'],validate='one_to_one').sort_values(['example_id','fingerprint']).reset_index(drop=True)
        z['fold']=[fold_by_key[(r.example_id,r.fingerprint)] for r in z.itertuples()]
        for validation in ['grouped','network']:
            # All controls use baseline and supplied n; Cunge additionally gets family ablations.
            names=list(variants) if method=='cunge' else ['geometry_15','geometry_15_plus_n']
            if validation=='network' and method!='cunge':continue # saved controls already cover network validation
            keys=range(5) if validation=='grouped' else sorted(z.example_id.unique())
            for response in ['shift_h','relative_shift_pct']:
                for name in names:
                    for fold in keys:
                        mask=z.fold.eq(fold) if validation=='grouped' else z.example_id.eq(fold)
                        tr,te=z[~mask],z[mask];cs=variants[name].copy()
                        if 'n_supplied' in cs and tr.n_supplied.nunique()==1:cs.remove('n_supplied')
                        fit=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=5,max_features=1.,random_state=20261005,n_jobs=1).fit(tr[cs],tr[response])
                        p=fit.predict(te[cs]);assert p.min()>=tr[response].min()-1e-10 and p.max()<=tr[response].max()+1e-10
                        part=te[['example_id','fingerprint','state_id','group','parent_lag_h','shift_h']].copy();part['validation']=validation;part['routing']=method;part['response']=response;part['features']=name;part['fold']=str(fold);part['observed']=te[response];part['predicted']=p;part['predicted_h']=p if response=='shift_h' else p*te.parent_lag_h.to_numpy()/100;part['training_mean']=tr[response].mean();predictions.append(part)
            print('Fit',method,validation,flush=True)
    p=pd.concat(predictions,ignore_index=True);p.to_parquet(OUT/'exploratory_predictions.parquet',index=False)
    rows=[];per_group=[]
    for keys,z in p.groupby(['validation','routing','response','features']):
        meta=dict(zip(['validation','routing','response','features'],keys))
        for units,yc,pc in [('target','observed','predicted'),('hours','shift_h','predicted_h')]:
            for scope,zz in [('pooled',z)]+list(z.groupby('example_id')):
                for weighting in ['pair','equal_group']:
                    w=None if weighting=='pair' else 1/zz.groupby('group').group.transform('size')
                    rows.append(dict(**meta,units=units,scope=scope,weighting=weighting,n=len(zz),groups=zz.group.nunique(),**metrics(zz[yc],zz[pc],w)))
            for group,zz in z.groupby('group'):per_group.append(dict(**meta,units=units,group=group,n=len(zz),**metrics(zz[yc],zz[pc])))
    pd.DataFrame(rows).to_csv(OUT/'exploratory_scores.csv',index=False);pd.DataFrame(per_group).to_csv(OUT/'per_group_scores.csv',index=False)
    # Replay the established network baseline to verify this independent implementation.
    previous=pd.read_parquet(PREVIOUS/'held_out_predictions.parquet')
    check=p[(p.validation=='network')&p.features.isin(['geometry_15','geometry_15_plus_n'])].merge(previous,on=['example_id','fingerprint','routing','response','features'],suffixes=('_new','_old'),validate='one_to_one')
    np.testing.assert_allclose(check.predicted_new,check.predicted_old,rtol=1e-11,atol=1e-12)

if __name__=='__main__':main()
