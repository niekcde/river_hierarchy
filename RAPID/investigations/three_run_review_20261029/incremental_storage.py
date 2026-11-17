"""Read-only hydraulic ΔJ calculation against completed real-network responses.

No routing, calibration, or model fitting. Input timings are used only as responses.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/Volumes/PhD/river_hierarchy/three_run_comparison_20261028/03_manning_midpoint')
OUT = ROOT/'RAPID/outputs/uncapped_event_window_20261020/three_run_review_20261029/incremental_storage'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    response_path = SOURCE/'responses.csv'
    receipt = json.loads((SOURCE/'completion.json').read_text())
    assert digest(response_path) == receipt['files'][str(response_path)]
    responses = pd.read_csv(response_path, dtype={'example': str})
    # Match the existing real-case audit; exclude the artificial single-centerline control.
    responses = responses[responses.state.ne('fully_collapsed')].copy()
    sources = {str(response_path): digest(response_path)}
    rows = []
    for example, group in responses.groupby('example'):
        for state in ['S000_base', *sorted(group.state.unique())]:
            path = SOURCE/'states'/example/state/'parameters.parquet'
            rec = json.loads((path.parent/'completion.json').read_text())
            sha = digest(path)
            assert sha == rec['files'][str(path)], str(path)
            sources[str(path)] = sha
            f = pd.read_parquet(path)
            columns = ['reference_flow_fraction','link_length_m','width','depth_m','slope','n_manning','reference_discharge_cms']
            assert np.isfinite(f[columns].to_numpy()).all() and (f[columns]>0).all().all()
            assert not f.rapid_celerity_capped.any()
            assert f.n_manning.nunique()==1
            qref = f.reference_discharge_cms/f.reference_flow_fraction
            np.testing.assert_allclose(qref, qref.iloc[0], rtol=1e-10)
            normal_h = (f.n_manning*f.reference_discharge_cms/(f.width*np.sqrt(f.slope)))**.6
            np.testing.assert_allclose(f.depth_m, normal_h, rtol=1e-10)
            edge_j = f.reference_flow_fraction*3*f.n_manning*f.link_length_m/(5*f.depth_m**(2/3)*np.sqrt(f.slope))
            j = float(edge_j.sum())
            volume = float((f.width*f.link_length_m*f.depth_m).sum())
            np.testing.assert_allclose(j, .6*volume/qref.iloc[0], rtol=1e-10)
            np.testing.assert_allclose(j, (f.reference_flow_fraction*f.link_length_m/f.rapid_celerity_mps).sum(), rtol=1e-10)
            rows.append(dict(example=example,state=state,J_seconds=j,V_m3=volume,n=float(f.n_manning.iloc[0]),Qref_cms=float(qref.iloc[0])))
        print('Reference storage calculated:', example, flush=True)
    states = pd.DataFrame(rows)
    parents = states[states.state.eq('S000_base')].drop(columns='state').rename(columns={c:'parent_'+c for c in states.columns if c not in ['example','state']})
    data = responses.merge(states,on=['example','state'],validate='many_to_one').merge(parents,on='example',validate='many_to_one')
    assert len(data)==len(responses) and not data.duplicated(['example','state','window']).any()
    np.testing.assert_allclose(data.n,data.parent_n,rtol=1e-12)
    np.testing.assert_allclose(data.Qref_cms,data.parent_Qref_cms,rtol=1e-12)
    data['delta_J_minutes']=(data.J_seconds-data.parent_J_seconds)/60
    data['centroid_shift_minutes']=data.centroid_shift_hours*60
    data['residual_minutes']=data.centroid_shift_minutes-data.delta_J_minutes
    records=[]
    for window,g in data.groupby('window'):
        for label,sub in [('all',g),*list(g.groupby('example'))]:
            y=sub.centroid_shift_minutes.to_numpy();e=sub.residual_minutes.to_numpy();den=((y-y.mean())**2).sum()
            records.append(dict(window=window,example=label,n=len(sub),r2_no_fit=1-(e**2).sum()/den if den>0 else np.nan,
                mae_minutes=abs(e).mean(),max_abs_error_minutes=abs(e).max(),bias_response_minus_J_minutes=e.mean()))
    OUT.mkdir(exist_ok=True,parents=True)
    states.to_csv(OUT/'state_storage.csv',index=False)
    data.to_parquet(OUT/'comparisons.parquet',index=False)
    scores=pd.DataFrame(records);scores.to_csv(OUT/'scores.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),method='fixed Manning midpoint reference; wide rectangular; constant fitted n per network, shared by parent and children',
        response='Saved finite-window excess-flow centroid, not a complete-event impulse centroid or observed collapse response',
        excluded='fully_collapsed (artificial single_centerline control)',comparisons_per_window=data.groupby('window').size().to_dict(),examples=data.example.nunique(),
        source_sha256=sources,script_sha256=digest(__file__)),indent=2)+'\n')
    print(scores[scores.example.eq('all')].to_string(index=False))
    return data

if __name__=='__main__':
    build()
