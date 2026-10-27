"""Audit current feature semantics and apply frozen forests to the Cunge pilot.

No training, imputation of undefined geometry, or production writes.
"""
from pathlib import Path
import sys, json, hashlib, importlib.util
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import cunge_pilot as pilot
ROOT=pilot.ROOT
REAL=ROOT/'RAPID/outputs/uncapped_event_window_20261020'
SC=ROOT/'RAPID/outputs/hydrograph_predictor_scopes_20261014'
OUT=ROOT/'synthetic_experiments/outputs/current_predictor_bridge_20261026'
SETTINGS=REAL/'current_model_comparison_20261023/settings.json'
sys.path.insert(0,str(ROOT/'RAPID/investigations/hydrograph_predictors_20261014'))
from build import descriptors, pair_features
from final_forest_search import predict
from rapid_tools.current_domains import allocation_tables, validate_domain
from hierarchy_level_definition.metrics.equivalent_resistance import solve_bubble_equivalent_K
from synthetic_experiments.networks import make_states, detect_units

SOURCES={}
def read(path):
    path=Path(path);SOURCES[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix=='.json':return json.loads(path.read_text())
    if path.suffix=='.parquet':return pd.read_parquet(path)
    return pd.read_csv(path,dtype={'example':str})


def resistance(frame):
    g=nx.MultiDiGraph()
    for r in frame.itertuples():g.add_edge(r.id_us_node,r.id_ds_node)
    source=next(n for n in g if g.in_degree(n)==0);sink=next(n for n in g if g.out_degree(n)==0)
    k=frame.link_length_m/(frame.width**2*frame.depth_m**3);scale=float(k.max())
    f=pd.DataFrame(dict(u=frame.id_us_node,v=frame.id_ds_node,K=k/scale))
    result=solve_bubble_equivalent_K(None,f,source,sink,smooth=False)
    if not result['solver_success']:raise ValueError(result['solver_message'])
    return float(result['K_eq'])*scale


def features(parent,child,selected_ids):
    # Descriptors and allocation are the same implementations as the real cases.
    p,c=descriptors(parent),descriptors(child)
    # Recognize exactly deterministic route lengths; never let floating arithmetic
    # turn an undefined 0/0 ratio into a spurious finite predictor.
    for m in [p,c]:
        if abs(m['route_length_max']-m['route_length_min'])<1e-10:
            m['route_length_sd']=0.
    row=pair_features(p,c,'B_')
    for key in ['route_length_mean','route_length_sd','width_mean','width_cv','route_entropy']:
        row['log1p__B_ratio_'+key]=float(np.log1p(row['B_ratio_'+key]))
    a=allocation_tables(parent)[1].set_index('id_link')
    if not set(selected_ids)<=set(a.index):raise ValueError('Selected unit outside B domain')
    row['domain_size']=float(a.loc[list(selected_ids),'mean_length_contribution_m'].sum()/1000)
    for side,f in [('parent',parent),('child',child)]:
        weights=allocation_tables(f)[1].set_index('id_link').loc[f.id_link,'mean_length_contribution_m'].to_numpy()
        np.testing.assert_allclose(weights.sum(),row['B_'+side+'_route_length_mean'],rtol=1e-10)
        row[side+'_G_inverse_depth']=float(np.sum(weights/(f.depth_m.to_numpy()*np.sqrt(f.slope.to_numpy()))))
        if side=='parent':row['parent_slope_slow']=float(np.average(1/np.sqrt(f.slope),weights=weights))
    row['delta_G_inverse_depth']=row['child_G_inverse_depth']-row['parent_G_inverse_depth']
    row['length_delta_km']=(c['route_length_mean']-p['route_length_mean'])/1000
    row['resistance_explicit_depth']=-np.log10(resistance(child)/resistance(parent))
    return row


def synthetic_frames(parameters,settings):
    graphs,pairs=make_states(parameters)
    depth=pilot.predict_based_depth(discharge_cms=settings['peak_cms'],width_m=parameters['total_width_m'],slope=parameters['reference_slope'])
    tables={};edgeids={}
    for state,g in graphs.items():
        nodes={n:i for i,n in enumerate(g.nodes)};rows=[];lookup={}
        for i,(u,v,k,d) in enumerate(g.edges(keys=True,data=True),1):
            lookup[u,v,k]=i
            rows.append(dict(id_link=i,id_us_node=nodes[u],id_ds_node=nodes[v],link_length_m=d['length_m'],width=d['w'],slope=d['slope'],depth_m=depth*(d['w']/parameters['total_width_m'])**settings['depth_alpha'],x0=g.nodes[u]['x'],x1=g.nodes[v]['x']))
        tables[state]=pd.DataFrame(rows);edgeids[state]=lookup
    return graphs,pairs,tables,edgeids


def synthetic_features(settings):
    rows=[];audit=[]
    for name,change in settings['cases'].items():
        pars=dict(settings['base_geometry'],**change)
        graphs,pairs,tables,edgeids=synthetic_frames(pars,settings)
        for pair in pairs:
            state=pair['parent_state'];g=graphs[state];units=detect_units(g)
            treatment=pair['child_state'].split('/')[1]
            all_edges=set().union(*(e for _,e in units))
            if treatment=='inner_only':selected=min((e for _,e in units),key=len)
            elif treatment in ['upstream_only','downstream_only']:
                ordered=sorted((e for _,e in units),key=lambda es:min(g.nodes[u]['x'] for u,v,k in es))
                selected=ordered[0 if treatment=='upstream_only' else 1]
            else:selected=all_edges
            # Main-island source unit is the outer unit: matches real source-unit
            # membership, not just physically removed island-boundary edges.
            a,b=.25*pars['total_length_m'],.75*pars['total_length_m']
            fs=[]
            for key in [state,pair['child_state']]:
                full=tables[key];part=full[full.x0.ge(a)&full.x1.le(b)]
                part,_=validate_domain(full,part);fs.append(part)
            ids={edgeids[state][e] for e in selected}
            row=features(*fs,ids)
            rows.append(dict(configuration=name,**pair,**row))
            audit.append(dict(configuration=name,child_state=pair['child_state'],B_scope='C' if pair['family']!='simple' else 'U',B_start_m=a,B_end_m=b,parent_domain_links=len(fs[0]),child_domain_links=len(fs[1]),selected_unit_links=len(ids),selected_ids=sorted(ids),island_size_convention='outer source-unit membership' if treatment=='main_island_only' else 'selected structural unit membership'))
    pilot.save(OUT/'domain_audit.json',audit)
    return pd.DataFrame(rows)


def real_parity(columns,training):
    meta=read(SC/'geometry_responses.parquet').set_index(['example','state'])
    graphs=read(SC/'domain_graphs.csv')
    jobs={}; geometry_paths={}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/routing_plan.json',REAL/'routing_plan.json']:
        for case in read(path)['cases']:
            for j in case['jobs']:
                if j['state_id']=='S000_base': geometry_paths.setdefault(case['example'],Path(j['parameters']))
                for key in {j['state_id'],*j.get('aliases',[])}:jobs[case['example'],key]=j
    # Small domains in every real network plus three containing-compound domains.
    candidates=training[training.cohort.eq('original')][['example','state']].merge(graphs,on=['example','state'])
    candidates=candidates[candidates.apply(lambda r:r.scope==meta.loc[(r.example,r.state),'B_scope'],axis=1)]
    selected=candidates.sort_values('parent_links').groupby('example').head(1)
    selected=pd.concat([selected,candidates[candidates.scope.eq('C')].sort_values('parent_links').head(3)]).drop_duplicates(['example','state'])
    records=[]
    for r in selected.itertuples():
        parent_path=Path(jobs[r.example,'S000_base']['parameters'])
        parent=read(parent_path);child=read(jobs[r.example,r.state]['parameters'])
        p,_=validate_domain(parent,read(SC/'graphs'/f'{r.parent_graph}.parquet'))
        c,_=validate_domain(child,read(SC/'graphs'/f'{r.child_graph}.parquet'))
        membership=read(geometry_paths[r.example].parents[3]/'variant/summary/unit_link_membership.csv')
        units=[int(x) for x in str(meta.loc[(r.example,r.state),'selected_unit_ids']).split(',')]
        ids=set(membership[membership.unit_id.isin(units)].id_link)
        f=features(p,c,ids);expected=training[(training.example==r.example)&(training.state==r.state)].iloc[0]
        for col in columns:
            np.testing.assert_allclose(f[col],expected[col],rtol=2e-6,atol=2e-7,err_msg=f'{r.example}/{r.state}/{col}')
            records.append(dict(example=r.example,state=r.state,scope=r.scope,feature=col,recomputed=f[col],saved=float(expected[col]),absolute_difference=abs(f[col]-expected[col])))
    pd.DataFrame(records).to_csv(OUT/'real_feature_parity.csv',index=False)
    return len(selected)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    settings=read(SETTINGS);cfg=read(pilot.CONFIG)
    completion=read(pilot.OUT/'completion.json');assert completion['numerical_screen_passed']
    training=read(REAL/'resistance_inverse_depth_panels_20261023/input.parquet')
    columns=list(dict.fromkeys(sum([v['columns'] for v in settings.values()],[])))
    count=real_parity(columns,training)
    f=synthetic_features(cfg)
    targets=read(pilot.OUT/'responses.csv');targets=targets[targets.threshold_fraction.eq(.005)]
    f=f.merge(targets[['configuration','child_state','centroid_shift_hours']],on=['configuration','child_state'],validate='one_to_one')
    f.to_parquet(OUT/'features.parquet',index=False)
    results=[];ranges=[]
    for name,spec in settings.items():
        path=Path(spec['artifact']);m=joblib.load(path)
        SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        expected=read(path.with_suffix('.json'));assert expected['sha256']==SOURCES[str(path)]
        assert m['columns']==spec['columns']
        eligible=np.isfinite(f[spec['columns']]).all(axis=1)
        pred=predict(m,f[eligible])
        for i,row in f.iterrows():
            missing=[c for c in spec['columns'] if not np.isfinite(row[c])]
            outside=[c for c in spec['columns'] if np.isfinite(row[c]) and not training[c].min()<=row[c]<=training[c].max()]
            results.append(dict(model=name,configuration=row.configuration,child_state=row.child_state,actual_hours=row.centroid_shift_hours,predicted_hours=float(pred[list(f[eligible].index).index(i)]) if eligible[i] else np.nan,status='predicted' if eligible[i] else 'undefined_feature_abstention',undefined_features=' | '.join(missing),outside_training_range=' | '.join(outside)))
        for col in spec['columns']:
            ranges.append(dict(model=name,feature=col,training_min=training[col].min(),training_max=training[col].max(),synthetic_min=f[col].min(),synthetic_max=f[col].max(),undefined=int((~np.isfinite(f[col])).sum())))
    result=pd.DataFrame(results);result['error_minutes']=60*(result.predicted_hours-result.actual_hours)
    result.to_csv(OUT/'predictions.csv',index=False);pd.DataFrame(ranges).to_csv(OUT/'feature_coverage.csv',index=False)
    scores=result[result.status.eq('predicted')].groupby('model').agg(n=('actual_hours','size'),mae_minutes=('error_minutes',lambda x:abs(x).mean()),max_abs_error_minutes=('error_minutes',lambda x:abs(x).max()))
    scores.to_csv(OUT/'pilot_errors.csv')
    for p in [Path(__file__),Path(pilot.__file__),ROOT/'RAPID/investigations/hydrograph_predictors_20261014/build.py',ROOT/'RAPID/investigations/parent_child_predictor_exploration/features.py',ROOT/'RAPID/src/rapid_tools/current_domains.py']:
        SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in SOURCES.items())
    pilot.save(OUT/'completion.json',dict(complete=True,real_parity_cases=count,pilot_pairs=len(f),prediction_rows=int(result.status.eq('predicted').sum()),abstention_rows=int(result.status.ne('predicted').sum()),fits=0,source_sha256=SOURCES,interpretation='Frozen all-data real models applied externally to synthetic pilot; no refitting. Undefined 0/0 geometry is not imputed. Range checks are marginal, not proof of joint support. Small selected pilot is not a representative validation set.'))
    print(scores.to_string());print(result[['model','configuration','child_state','status','error_minutes']].to_string(index=False))

if __name__=='__main__':main()
