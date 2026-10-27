"""Isolated saved-output audit. Never routes, calibrates, or edits production data."""
from pathlib import Path
from datetime import datetime
import sys, json, hashlib
import numpy as np
import pandas as pd
import networkx as nx
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'RAPID/investigations/parent_child_predictor_exploration'))
from features import describe, METRICS, HYDRAULIC
OUT = ROOT/'RAPID/outputs'/('geometry_response_foundation_'+datetime.now().strftime('%Y%m%dT%H%M%S'))
OUT.mkdir(exist_ok=False)
hashes = {}

def read(path):
    path = Path(path)
    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.suffix == '.parquet': return pd.read_parquet(path)
    if path.suffix == '.json': return json.loads(path.read_text())
    return pd.read_csv(path)

def ids(value):
    if pd.isna(value): return set()
    return {int(float(x)) for x in str(value).split(',') if x.strip()}

def entropic_branches(frame):
    """Proposed visit-weighted local eBI analogue, NOT transect-based eBI."""
    graph=nx.DiGraph(); outgoing={}
    for r in frame.itertuples():
        graph.add_edge(r.id_us_node,r.id_ds_node)
        outgoing.setdefault(r.id_us_node,[]).append((r.id_ds_node,r.width))
    order=list(nx.topological_sort(graph)); visit={order[0]:1.}; vals=[]; weights=[]
    for u in order:
        ee=outgoing.get(u,[])
        if not ee: continue
        p=np.array([w for v,w in ee],float);p/=p.sum()
        if len(p)>1: vals.append(float(np.exp(-p@np.log(p))));weights.append(visit[u])
        for (v,w),prob in zip(ee,p): visit[v]=visit.get(v,0.)+visit[u]*prob
    if not vals: return dict(entropic_branches_mean=np.nan,entropic_branches_sd=np.nan,has_splits=False)
    mu=float(np.average(vals,weights=weights))
    return dict(entropic_branches_mean=mu,entropic_branches_sd=float(np.sqrt(np.average((np.array(vals)-mu)**2,weights=weights))),has_splits=True)

def peak_diagnostics(wave):
    q = wave.q_outlet_cms.to_numpy()
    t = pd.to_datetime(wave.time_utc).astype('int64').to_numpy()/1e9
    assert np.allclose(np.diff(t), 60) and np.isfinite(q).all()
    i = int(q.argmax()); offset = np.nan
    ties = np.flatnonzero(abs(q-q[i]) <= max(np.ptp(q)*1e-10, 1e-9))
    if 0 < i < len(q)-1 and len(ties) == 1:
        den = q[i-1]-2*q[i]+q[i+1]
        if den < 0: offset = .5*(q[i-1]-q[i+1])/den*60
    return dict(peak_utc=str(wave.time_utc.iloc[i]), peak_cms=q[i],
                quadratic_offset_seconds=offset, exact_top_samples=len(ties),
                first_cms=q[0], last_cms=q[-1])

review = next(r for r in read(ROOT/'RAPID/outputs/bounded_beta_calibration_20261009/timing_targets.json')['records'] if r['example']=='54')
span = (pd.Timestamp(review['crest_end_utc'])-pd.Timestamp(review['crest_start_utc'])).total_seconds()
separation = max(60., int(span/180.)*60.)

def lobes(wave):
    q = wave.q_outlet_cms.to_numpy()
    t = pd.to_datetime(wave.time_utc).astype('int64').to_numpy()/1e9
    peaks, _ = find_peaks(q, prominence=max(np.ptp(q)*1e-4, 1e-8), distance=int(np.ceil(separation/60)))
    peaks = peaks[q[peaks] >= q.min()+.8*np.ptp(q)]
    best = np.sort(peaks[np.argsort(q[peaks])[-2:]])
    if len(best) != 2: return dict(status='merged_or_unresolved')
    a,b = best
    return dict(status='two_lobes', left_seconds=t[a], right_seconds=t[b], center_seconds=(t[a]+t[b])/2,
                left_utc=str(wave.time_utc.iloc[a]), right_utc=str(wave.time_utc.iloc[b]),
                left_cms=q[a], right_cms=q[b], separation_minutes=(t[b]-t[a])/60)

first = ROOT/'RAPID/outputs/predictor_release_review_20261012T104056/canonical_child_first_look.csv'
pairs = read(first); pairs['example'] = pairs.example.astype(str).str.zfill(2)
members=[]; zero=[]; lobe_rows=[]; geometry=[]; diagnostics=[]; states=[]; aliases=[]; base54=None
for ex in sorted(pairs.example.unique()):
    root = ROOT/f'network_variants/output/{ex}_indep__anchored'
    units = read(root/'states/S000_base/hierarchy/unit_summary.csv').set_index('unit_id')
    regs=[]; selections={}
    for fam in ['indep','compound_bubbles','island']:
        folder=ROOT/f'network_variants/output/{ex}_{fam}__anchored'
        if not (folder/'rapid_run_registry.csv').exists(): continue
        reg=read(folder/'rapid_run_registry.csv');reg['family']=fam;regs.append(reg)
        if fam=='island':
            manifest=read(folder/'island_analysis_manifest.csv')
            for r in manifest.itertuples():
                selections[(fam,r.island_id)]=ids(r.current_unit_ids)|ids(r.source_unit_id)
        else:
            manifest=read(folder/'state_registry.csv')
            assert manifest.parent_state_id.dropna().eq('S000_base').all()
            for r in manifest.itertuples(): selections[(fam,r.state_id)]=ids(r.selected_unit_ids)
    regs=pd.concat(regs,ignore_index=True)
    base=regs[regs.state_id.eq('S000_base')].iloc[0]
    bw=read(base.outlet_parquet);bd=peak_diagnostics(bw)
    bl=lobes(bw) if ex=='54' else None
    if ex=='54': base54=bl
    bp=read(base.parameters_parquet)
    gm,meta=describe(bp,False); assert meta['status']=='ok'
    base_geom={k:float(v) for k,v in gm.items() if k not in HYDRAULIC}
    for fp,g in regs.groupby('fingerprint',sort=False):
        r=g.iloc[0]; isbase=fp==base.fingerprint
        assert g.outlet_parquet.nunique()==1
        selected=set()
        for a in g.itertuples():
            uu=selections.get((a.family,a.state_id),set()); selected |= uu
            aliases.append(dict(example=ex,fingerprint=fp,family=a.family,state_id=a.state_id,unit_ids=','.join(map(str,sorted(uu)))))
        assert selected <= set(units.index), (ex,selected-set(units.index))
        us=units.loc[sorted(selected)]
        member_bubbles=sorted(set(us.loc[us.in_compound_bubble.astype(bool),'compound_bubble_id'].dropna().astype(int)))
        members.append(dict(example=ex,fingerprint=fp,state_id=r.state_id,is_base=isbase,
            unit_ids=','.join(map(str,sorted(selected))),membership_resolved=bool(selected) or isbase,
            any_compound_member=bool(us.in_compound_bubble.any()),any_nested_unit=bool(us.is_compound.any()),
            any_complex_class=bool(us['class'].eq('compound_or_nested_complex').any()),
            compound_bubble_ids=','.join(map(str,member_bubbles)),
            unit_classes=';'.join(sorted(us['class'].unique())),families=','.join(sorted(g.family.unique()))))
        states.append(dict(example=ex,fingerprint=fp,is_base=isbase,state_id=r.state_id,
            parent_fingerprint=None if isbase else base.fingerprint,parameters_parquet=r.parameters_parquet,
            outlet_parquet=r.outlet_parquet,forcing_parquet=r.forcing_parquet,result_json=r.result_json,
            routing_dt_seconds=r.routing_dt_seconds,forcing_dt_seconds=r.forcing_dt_seconds))
        f=bp if isbase else read(r.parameters_parquet)
        m,meta=describe(f,False);assert meta['status']=='ok'
        values={k:float(v) for k,v in m.items() if k not in HYDRAULIC}
        geometry.append(dict(example=ex,fingerprint=fp,scope='whole_network',**values,**entropic_branches(f)))
        diagnostics.append(dict(example=ex,fingerprint=fp,n_manning=float(f.n_manning.iloc[0]),
             beta=float(f.diffusion_multiplier.iloc[0]),depth_mean_m=float(np.average(f.depth_m,weights=f.link_length_m)),
             celerity_mean_mps=float(np.average(f.rapid_celerity_mps,weights=f.link_length_m)),
             kb_mean=float(np.average(f.kb_local,weights=f.link_length_m))))
        if isbase: continue
        pr=pairs[(pairs.example==ex)&(pairs.fingerprint==fp)].iloc[0]
        if pr.sampled_peak_shift_minutes==0 or ex=='54':
            w=read(r.outlet_parquet);d=peak_diagnostics(w)
            assert len(w)==len(bw) and w.time_utc.equals(bw.time_utc)
            if pr.sampled_peak_shift_minutes==0:
                zero.append(dict(example=ex,fingerprint=fp,state_id=r.state_id,
                     sampled_shift_minutes=0,feature_shift_minutes=pr.feature_shift_minutes,
                     quadratic_shift_seconds=d['quadratic_offset_seconds']-bd['quadratic_offset_seconds'],
                     base_exact_top_samples=bd['exact_top_samples'],child_exact_top_samples=d['exact_top_samples'],
                     peak_change_pct=pr.peak_total_change_pct,peak_utc=d['peak_utc']))
            if ex=='54':
                l=lobes(w);assert l['status']=='two_lobes'
                assert abs(l['center_seconds']-pd.Timestamp(r.calibration_feature_utc).timestamp())<1e-5
                lobe_rows.append(dict(example=ex,fingerprint=fp,state_id=r.state_id,
                    **l,**{k+'_shift_minutes':(l[k+'_seconds']-bl[k+'_seconds'])/60 for k in ['left','right','center']},
                    sampled_peak_shift_minutes=pr.sampled_peak_shift_minutes,
                    left_change_pct=100*(l['left_cms']/bl['left_cms']-1),right_change_pct=100*(l['right_cms']/bl['right_cms']-1)))
    print(ex,'done',flush=True)

membership=pd.DataFrame(members)
pairs=pairs.merge(membership.drop(columns=['state_id','is_base']),on=['example','fingerprint'],validate='one_to_one')
state_table=pd.DataFrame(states); assert len(state_table)==2187 and len(pairs)==2172
assert pairs.membership_resolved.all()
state_table.to_parquet(OUT/'canonical_states.parquet',index=False)
pairs.to_parquet(OUT/'canonical_pairs.parquet',index=False)
pd.DataFrame(aliases).to_csv(OUT/'state_aliases.csv',index=False)
pd.DataFrame(zero).to_csv(OUT/'same_sampled_peak_audit.csv',index=False)
pd.DataFrame(lobe_rows).to_csv(OUT/'54_lobe_audit.csv',index=False)
pd.DataFrame(geometry).to_parquet(OUT/'network_geometry.parquet',index=False)
pd.DataFrame(diagnostics).to_parquet(OUT/'model_diagnostics_NOT_predictors.parquet',index=False)
pairs[pairs.sampled_peak_shift_minutes>0].to_csv(OUT/'later_sampled_peaks_membership.csv',index=False)
pd.DataFrame([dict(name=k,length_exponent=v[0],time_exponent=v[1],family=v[2],description=v[3],
    computed_scope='whole_network',allocation='local outgoing width fractions only',
    local_domain_status='pending boundary/physical-overlap audit') for k,v in METRICS.items() if k not in HYDRAULIC]).to_csv(OUT/'computed_geometry_catalog.csv',index=False)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
(OUT/'provenance.json').write_text(json.dumps(dict(source_hashes=hashes,source_unchanged=True,
    scope='Saved-data audit; geometry uses physical lengths and widths only. No rerouting or fitting.',
    qualifications=['Membership is unit-summary classification, not yet physical-overlap CV grouping.',
    'Whole-network geometry computed; matched local and containing-compound boundaries remain to audit.',
    'Three-point quadratic timing is a sensitivity diagnostic, not a recovered exact peak or confidence interval.',
    'Response volume/shape definitions require an event window and a defensible baseline; no initial-flow clipping.',
    '54 two-lobe selection exactly follows the production feature operator; ordered lobes are not separately recalibrated.'],
    base54_lobes=base54),indent=2))
print('OUT',OUT)
print(pairs[pairs.sampled_peak_shift_minutes>0].groupby(['any_compound_member','any_nested_unit','any_complex_class']).size())
