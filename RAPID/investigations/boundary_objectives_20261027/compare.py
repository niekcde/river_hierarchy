"""Isolated joint boundary-head objective comparison; no production writers/routing."""
from pathlib import Path
import sys,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'RAPID/investigations/member_boundary_review_20261027')]
import numpy as np,pandas as pd
from feasibility import head_series,path_constraints
from network_variants.reviewed_geometry import ReviewedGeometry
from solver import quadratic_projection
OUT=ROOT/'RAPID/outputs/boundary_objectives_20261027_v2';BASE=ROOT/'RAPID/assets/bounded_robust_slopes_20261002'
CANDIDATES=Path('/tmp/member_boundary_candidate_results_20261027_v2.json')


def main():
    OUT.mkdir(exist_ok=False);sources={};cache={};case_rows=[];edge_rows=[];node_rows=[];certificates=[];case_meta=[]
    def track(p):
        p=str(p)
        if p not in sources:sources[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
        return Path(p)
    def read(p):return pd.read_parquet(track(p))
    candidates=json.loads(track(CANDIDATES).read_text());shutil.copy2(CANDIDATES,OUT/'candidate_inventory.json')
    jobs={}
    for path in [ROOT/'RAPID/outputs/event_window_plan_20261013/plan.json',ROOT/'RAPID/outputs/hydrograph_closed_unit_extension_20261019/routing_plan.json',ROOT/'RAPID/outputs/uncapped_event_window_20261020/routing_plan.json']:
        for c in json.loads(track(path).read_text())['cases']:
            for j in c['jobs']:jobs[c['example'],j['state_id']]=j
    release=ReviewedGeometry();track(release.manifest)
    chosen=[z for z in candidates if z['minimum_released_nodes']==1]
    for num,item in enumerate(chosen,1):
        ex,state=item['example'],item['state']
        if ex not in cache:
            parent=read(release.lookup(ex,'indep','S000_base')['original_frame']);heads=head_series(parent);old=read(BASE/ex/'floor_0.05/old/links.parquet').set_index('id_link');routing=read(jobs[ex,'S000_base']['parameters']).set_index('id_link');cache[ex]=parent,heads,old,routing
        parent,heads,old,routing=cache[ex];pidx=parent.set_index('id_link');record=release.lookup(ex,'indep',state);contract=json.loads(track(record['contract_path']).read_text());f=read(record['original_frame']).reset_index(drop=True);selected=set(contract['selected_parent_links']);boundary=set(contract['boundary_nodes']);ext=parent[~parent.id_link.isin(selected)];fixed=set(ext.u)|set(ext.v)|boundary
        pathfile=Path(record.get('source_bundle') or '')/'exterior_paths.json'
        if record.get('source_bundle') and pathfile.exists():
            paths=json.loads(track(pathfile).read_text());assert all(len(p['child_links'])==1 for p in paths)
            outside={p['child_links'][0]:p['parent_link'] for p in paths}
        else:
            groups={}
            for z in f.itertuples():groups.setdefault((z.u,z.v),[]).append((z.id_link,z.link_length_m))
            outside={}
            for z in ext.itertuples():
                hits=[lid for lid,L in groups.get((z.u,z.v),[]) if np.isclose(L,z.link_length_m,rtol=1e-8,atol=1e-5)];outside[z.id_link if z.id_link in hits else hits[0]]=z.id_link
        floor=float(item['compound_floor']);floors=np.array([min(old.loc[outside[z.id_link],'lower_bound'],pidx.loc[outside[z.id_link],'slope']) if z.id_link in outside else floor for z in f.itertuples()]);floors=np.nan_to_num(floors,nan=0.)
        default_drops=np.maximum(floors*f.link_length_m.to_numpy(),1e-10)
        original_paths=path_constraints(f,heads.loc[sorted(fixed)],default_drops)
        if item['previous_status']=='nonpositive_fixed_head_path':movable=sorted(item['candidate_nodes'])
        else:movable=sorted({v for path in original_paths if path['available_drop_m']<path['path_weight']-1e-9 for v in [path['upstream'],path['downstream']]})
        assert set(movable)<=boundary and 1<=len(movable)<=3
        ix={v:i for i,v in enumerate(movable)};affected=f[f.id_link.isin(outside)&(f.u.isin(movable)|f.v.isin(movable))].copy();parent_ids=[outside[lid] for lid in affected.id_link];pr=pidx.loc[parent_ids];rr=routing.loc[parent_ids]
        D0=(pr.head_us-pr.head_ds).to_numpy();L0=pr.link_length_m.to_numpy();LC=affected.link_length_m.to_numpy();Sref=pr.slope.to_numpy();C=np.zeros((len(affected),len(movable)))
        for j,z in enumerate(affected.itertuples()):
            for v,sgn in [(z.u,1),(z.v,-1)]:
                if v in ix:C[j,ix[v]]+=sgn
        geom_changed=int((~np.isclose(LC,L0,rtol=1e-8,atol=1e-5)).sum())
        case_meta.append(dict(example=ex,state=state,candidates=movable,affected_exterior_links=len(affected),affected_exterior_length_changes=geom_changed,compound_floor=floor))
        for length_policy in ['saved_geometry','reference_exterior_lengths']:
            L=LC if length_policy=='saved_geometry' else L0
            lengths=f.link_length_m.to_numpy().copy()
            if length_policy=='reference_exterior_lengths':
                for j,z in enumerate(f.itertuples()):
                    if z.id_link in outside:lengths[j]=pidx.loc[outside[z.id_link],'link_length_m']
            paths=path_constraints(f,heads.loc[sorted(fixed)],np.maximum(floors*lengths,1e-10));A=[];b=[];constraints=[]
            for path in paths:
                row=np.zeros(len(movable))
                for v,sgn in [(path['upstream'],1),(path['downstream'],-1)]:
                    if v in ix:row[ix[v]]+=sgn
                rhs=path['path_weight']-path['available_drop_m']
                if not np.any(row):
                    if rhs>1e-8:raise ValueError('Unrelated fixed exterior constraint violated')
                else:A.append(row);b.append(rhs);constraints.append(path)
            A=np.array(A);b=np.array(b);scale=max(float(np.max(b)),.001)
            for objective in ['absolute_drop','relative_drop','absolute_slope']:
                weights=np.ones(len(affected)) if objective=='absolute_drop' else 1/D0 if objective=='relative_drop' else 1/L
                target=np.zeros(len(affected)) if objective!='absolute_slope' else Sref*L-D0
                x,qa=quadratic_projection(C,target/scale,weights,A,b/scale);delta=x*scale;change=C@delta;newdrop=D0+change;S=newdrop/L;Scontrol=D0/L
                if (S<=0).any():raise ValueError('Nonpositive exterior slope')
                assert np.max(b-A@delta)<1e-8
                # Slope-only fixed-depth response diagnostics. Keep parent widths,
                # depths, Q_ref, n, beta and cap settings across all objectives.
                factor=(5/3)*rr.depth_m.to_numpy()**(2/3)/rr.n_manning.to_numpy()
                def celerity(sl):return np.clip(factor*np.sqrt(sl),rr.celerity_min_mps.to_numpy(),rr.celerity_max_mps.to_numpy())
                cref=celerity(Sref);c0=celerity(Scontrol);c=celerity(S)
                Tref=L0/cref;Tcontrol=L/c0;T=L/c
                diff_factor=rr.reference_discharge_cms.to_numpy()/(2*rr.width.to_numpy());Fref=diff_factor/Sref;Fcontrol=diff_factor/Scontrol;F=diff_factor/S
                flow_weight=rr.reference_flow_fraction.to_numpy();rel=S/Sref-1;adjustment_rel=S/Scontrol-1
                key=dict(example=ex,state=state,length_policy=length_policy,objective=objective)
                score=dict(**key,candidate_nodes=len(movable),changed_boundary_nodes=int((abs(delta)>1e-8).sum()),affected_exterior_links=len(affected),changed_exterior_drops=int((abs(change)>1e-8).sum()),rms_head_drop_change_m=float(np.sqrt(np.mean(change**2))),max_head_change_m=float(abs(delta).max()),rms_relative_drop_change=float(np.sqrt(np.mean((change/D0)**2))),rms_absolute_slope_change=float(np.sqrt(np.mean((S-Sref)**2))),max_relative_slope_change=float(abs(rel).max()),rms_relative_slope_change=float(np.sqrt(np.mean(rel**2))),max_adjustment_relative_slope_change=float(abs(adjustment_rel).max()),weighted_signed_travel_change_s=float(flow_weight@(T-Tref)),weighted_absolute_travel_change_s=float(flow_weight@abs(T-Tref)),weighted_signed_adjustment_travel_s=float(flow_weight@(T-Tcontrol)),weighted_absolute_adjustment_travel_s=float(flow_weight@abs(T-Tcontrol)),max_relative_diffusivity_change=float(abs(F/Fref-1).max()),max_adjustment_relative_diffusivity_change=float(abs(F/Fcontrol-1).max()),capped_exterior_links=int((~np.isclose(c,factor*np.sqrt(S),rtol=1e-12,atol=1e-12)).sum()),primal_residual_m=qa['primal_residual']*scale,stationarity=qa['stationarity'])
                case_rows.append(score)
                for v,d in zip(movable,delta):node_rows.append(dict(**key,node=v,reference_head_m=float(heads.loc[v]),head_change_m=float(d),new_head_m=float(heads.loc[v]+d)))
                for j,z in enumerate(affected.itertuples()):edge_rows.append(dict(**key,parent_link=int(parent_ids[j]),child_link=int(z.id_link),u=int(z.u),v=int(z.v),parent_length_m=float(L0[j]),saved_child_length_m=float(LC[j]),evaluation_length_m=float(L[j]),reference_drop_m=float(D0[j]),head_drop_change_m=float(change[j]),reference_slope=float(Sref[j]),new_slope=float(S[j]),adjustment_slope_change=float(S[j]-Scontrol[j]),relative_slope_change=float(rel[j]),reference_celerity_mps=float(cref[j]),new_celerity_mps=float(c[j]),reference_travel_seconds=float(Tref[j]),new_travel_seconds=float(T[j]),adjustment_travel_seconds=float(T[j]-Tcontrol[j]),reference_diffusivity_m2_s=float(Fref[j]),new_diffusivity_m2_s=float(F[j]),reference_flow_fraction=float(flow_weight[j])))
                certificates.append(dict(**key,head_scale_m=scale,**qa,active_paths=[constraints[i] for i in qa['active_constraints']]))
        if num%10==0:print('Completed',num,'/',len(chosen),flush=True)
    pd.DataFrame(case_rows).to_csv(OUT/'case_metrics.csv',index=False);pd.DataFrame(edge_rows).to_parquet(OUT/'exterior_reaches.parquet',index=False);pd.DataFrame(node_rows).to_csv(OUT/'boundary_heads.csv',index=False)
    (OUT/'case_metadata.json').write_text(json.dumps(case_meta,indent=2));(OUT/'optimality_certificates.json').write_text(json.dumps(certificates,indent=2))
    for path,h in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
    (OUT/'manifest.json').write_text(json.dumps(dict(complete=True,cases=len(chosen),objectives=3,length_policies=2,quadratic_programs=len(case_rows),source_hashes=sources,production_modified=False,routing_runs=0,internal_head_objective_not_selected=True,reference_floor_note='Exterior floor is capped at its accepted reference slope solely to retain existing micrometre-scale reference-solver residuals; new replacement reaches use the frozen compound floor.'),indent=2));print('Complete',len(case_rows),'certified optima')

if __name__=='__main__':main()
