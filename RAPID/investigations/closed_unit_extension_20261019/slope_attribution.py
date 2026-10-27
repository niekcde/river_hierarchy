"""Geometry-only slope descriptors and diagnostic speed-factor attribution; no fits."""
import itertools,math,json
import numpy as np
import pandas as pd
from geometry_audit import OUT,pilot
DEST=OUT/'slope_attribution';FACTORS=['slope','depth','roughness','cap']

def value(frame,mask,reference):
    length=frame.reference_flow_fraction.to_numpy()*frame.link_length_m.to_numpy();w=length/length.sum()
    S=frame.slope.to_numpy() if mask&1 else reference['slope']
    h=frame.depth_m.to_numpy() if mask&2 else reference['depth']
    n=frame.n_manning.to_numpy() if mask&4 else reference['roughness']
    c=(5/3)*h**(2/3)*np.sqrt(S)/n
    if mask&8:c=np.clip(c,frame.celerity_min_mps.to_numpy(),frame.celerity_max_mps.to_numpy())
    return float(np.log(1/np.sum(w/c)))

def attribute(frame,reference):
    v={m:value(frame,m,reference) for m in range(16)};result={}
    for j,name in enumerate(FACTORS):
        phi=0.
        for m in range(16):
            if m&(1<<j):continue
            k=m.bit_count();weight=math.factorial(k)*math.factorial(3-k)/math.factorial(4)
            phi+=weight*(v[m|(1<<j)]-v[m])
        result[name]=phi
    assert np.isclose(sum(result.values()),v[15]-v[0],atol=1e-12)
    actual=(frame.reference_flow_fraction*frame.link_length_m).sum()/(frame.reference_flow_fraction*frame.link_length_m/frame.rapid_celerity_mps).sum()
    assert np.isclose(np.exp(v[15]),actual,rtol=1e-10)
    return result,actual

def descriptor(f):
    length=f.reference_flow_fraction*f.link_length_m;w=length/length.sum();S=f.slope.to_numpy();proxy=float((length/np.sqrt(S)).sum()/1000)
    mean=float((w*S).sum());effective=float((w/np.sqrt(S)).sum()**-2)
    return dict(length_km=float(length.sum()/1000),slope_mean=mean,slope_effective=effective,slope_heterogeneity=mean/effective,slope_proxy_km=proxy,depth_geomean=float(np.exp((w*np.log(f.depth_m)).sum())),roughness=float(f.n_manning.iloc[0]))

def main():
 DEST.mkdir(exist_ok=True);plan=pilot.read(OUT/'routing_plan.json');rows=[];attributes=[];provenance=[]
 refs=[dict(slope=1e-4,depth=5.,roughness=.03),dict(slope=2e-4,depth=10.,roughness=.02)]
 for case in plan['cases']:
  ex=case['example'];jobs={r['state_id']:r for r in case['jobs']};p=pilot.read(jobs['S000_base']['parameters']);c=pilot.read(jobs['all_closed_units_collapsed']['parameters']);line=pilot.read(OUT/f'variants/{ex}/all_closed_units_collapsed/lineage.csv');selected=line[line.role.eq('collapsed_block')]
  canonical=pilot.Path(jobs['S000_base']['parameters']).parents[3];mp=canonical/'hydraulic_context/hydraulic_context_manifest.json';meta=pilot.read(mp);source=pilot.read(meta['provenance']['source_frame'])
  cols=['slope','head_us','head_ds','width','depth_m'];a=p.set_index('id_link')[cols].sort_index();s=source.set_index('id_link')[cols].sort_index();pd.testing.assert_index_equal(a.index,s.index);np.testing.assert_allclose(a,s,rtol=1e-10,atol=1e-8)
  for side,f,ids in [('parent',p,selected.parent_link_id),('child',c,selected.child_link_id)]:
   f=f[f.id_link.isin(ids)].copy();ratio=f.depth_m/f.width**.65;assert np.allclose(ratio,ratio.iloc[0],rtol=1e-10)
   np.testing.assert_allclose((f.head_us-f.head_ds)/f.link_length_m,f.slope,rtol=1e-10)
   np.testing.assert_allclose((5/3)*f.depth_m**(2/3)*np.sqrt(f.slope)/f.n_manning,f.rapid_celerity_mps_raw,rtol=1e-10)
   rows.append(dict(example=ex,side=side,**descriptor(f)))
   for ref_id,ref in enumerate(refs):
    phi,speed=attribute(f,ref);attributes.append(dict(example=ex,side=side,reference=ref_id,effective_speed=speed,**phi))
  provenance.append(dict(example=ex,manifest=str(mp),source_frame=meta['provenance']['source_frame'],source_method=meta.get('section_slope_summary',{}).get('source_method'),source_matches_active=True,depth_width_exponent_verified=.65))
 d=pd.DataFrame(rows);d.to_csv(DEST/'slope_descriptors.csv',index=False);a=pd.DataFrame(attributes);a.to_csv(DEST/'speed_factor_attributions.csv',index=False);pd.DataFrame(provenance).to_csv(DEST/'provenance.csv',index=False)
 contrasts=[]
 for ref in [0,1]:
  for side in ['parent','child']:
   z=a[(a.reference==ref)&(a.side==side)].set_index('example')
   for ex in ['41','46','55']:
    vals={k:float(z.loc['22',k]-z.loc[ex,k]) for k in FACTORS};logratio=np.log(z.loc['22','effective_speed']/z.loc[ex,'effective_speed']);assert np.isclose(sum(vals.values()),logratio)
    contrasts.append(dict(reference=ref,side=side,comparison='22/'+ex,log_speed_ratio=logratio,speed_ratio=np.exp(logratio),**vals))
 pd.DataFrame(contrasts).to_csv(DEST/'speed_contrasts.csv',index=False)
 features=[];responses=pilot.read(OUT/'responses.parquet');r=responses[(responses.window=='0.5%')&responses.cohort.eq('segmented')].set_index('example')
 for ex,g in d.groupby('example'):
  z=g.set_index('side');features.append(dict(example=ex,centroid_shift_hours=r.loc[ex].centroid_shift_hours,parent_length_km=z.loc['parent','length_km'],delta_length_km=z.loc['child','length_km']-z.loc['parent','length_km'],parent_slope_effective=z.loc['parent','slope_effective'],slope_effective_ratio=z.loc['child','slope_effective']/z.loc['parent','slope_effective'],delta_slope_proxy_km=z.loc['child','slope_proxy_km']-z.loc['parent','slope_proxy_km']))
 q=pd.DataFrame(features);q.to_csv(DEST/'segmented_slope_candidates.csv',index=False)
 cor=[]
 for col in q.columns[2:]:cor.append(dict(predictor=col,pearson=q[col].corr(q.centroid_shift_hours),spearman=q[col].corr(q.centroid_shift_hours,method='spearman'),n=len(q)))
 pd.DataFrame(cor).to_csv(DEST/'descriptive_associations.csv',index=False)
 pilot.save(DEST/'manifest.json',dict(sources=pilot.SOURCES,code_sha256=pilot.sha(__file__),reference_scenarios=refs,scope='Selected edited footprints, same parent/child allocation as segmented predictors, connectors excluded',attribution='Exact four-factor Shapley decomposition of log effective speed against uniform S,h,n and no cap; mathematical counterfactuals, not causal independent interventions or RF SHAP',warning='15-case descriptive analysis; no predictor selection significance, no validation performance, no changes to frozen model'))
 print(pd.DataFrame(contrasts).query("reference==0 and side=='parent'").round(4).to_string(index=False));print(d[d.example.isin(['22','41','46','55'])].round(6).to_string(index=False));print(pd.DataFrame(cor).round(4).to_string(index=False))
if __name__=='__main__':main()
