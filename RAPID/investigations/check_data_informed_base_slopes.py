"""Independent artifact checks and converged-objective profile controls."""
from run_data_informed_base_slopes import *
from rapid_tools.reference_profile import _curvature_penalty
controls=[];checks=[]
saved_controls=pd.read_csv(OUT/'profile_controls.csv',dtype={'example':str}) if '--reuse-controls' in sys.argv else None
for ex,p in sources.items():
 out=OUT/ex
 if not (out/'summary.json').exists():continue
 pp=pd.read_parquet(out/'reference_profile.parquet');nn=pd.read_parquet(out/'candidate_node_hydraulic_context.parquet');ll=pd.read_parquet(out/'candidate_link_hydraulic_context.parquet')
 keep=['id_link','id_us_node','id_ds_node','actual_length_m','ownership_kind','ownership_domain_id','slope_used','previous_slope_used','candidate_head_us_m','candidate_head_ds_m','candidate_slope_m_per_m','slope_source_method','profile_head_us_m','profile_head_ds_m','profile_direct_slope_m_per_m']
 ll=ll[keep];nn=nn[['id_node','reference_section_dist_out_m','reference_section_dist_source','profile_head_m','candidate_head_m']]
 dx=np.diff(pp.reference_section_dist_out_m);dh=np.diff(pp.fitted_wse_m);floor=pp.minimum_interval_slope.to_numpy()[1:]
 pp=pp.drop(columns=[c for c in pp if c.startswith('manual_profile_adjustment')],errors='ignore');pp['floor_active']=np.r_[False,np.isclose(dh,floor*dx,atol=2e-5,rtol=1e-7)]
 pp.to_parquet(out/'reference_profile.parquet',index=False);nn.to_parquet(out/'candidate_node_hydraulic_context.parquet',index=False);ll.to_parquet(out/'candidate_link_hydraulic_context.parquet',index=False)
 heads=nn.set_index('id_node').candidate_head_m;err=ll.slope_used*ll.actual_length_m-(ll.id_us_node.map(heads)-ll.id_ds_node.map(heads))
 assert abs(err).max()<1e-10 and (ll.slope_used>0).all()
 check=dict(example=ex,n_links=len(ll),max_link_head_residual_m=float(abs(err).max()),profile_floor_violation_m=float(max(0,np.max(floor*dx-dh))))
 check['max_shared_node_head_error_m']=float(max(abs(ll.id_us_node.map(heads)-ll.candidate_head_us_m).max(),abs(ll.id_ds_node.map(heads)-ll.candidate_head_ds_m).max()))
 checks.append(check)
 if saved_controls is not None:
  controls.extend(saved_controls[saved_controls.example.str.zfill(2)==ex].to_dict('records'))
  continue
 p=Path(p);f=pd.read_parquet(p/'reference_profile.parquet');o=pd.read_parquet(p/'reference_profile_observations.parquet');audit=json.loads((p/'reference_profile_audit.json').read_text())['summary']
 x=f.reference_section_dist_out_m.to_numpy();n=len(x);acqs=pd.Index(sorted(o.acquisition_id.unique()));ai=acqs.get_indexer(o.acquisition_id);ni=np.searchsorted(x,o.reference_section_dist_out_m);N=len(o);nv=n+len(acqs)
 mat=sparse.coo_matrix((np.ones(2*N),(np.r_[np.arange(N),np.arange(N)],np.r_[ni,n+ai])),shape=(N,nv)).tocsr();center=f.fitted_wse_m.median();y=o.wse_m.to_numpy()-center
 penalty=sparse.block_diag((2*float(audit['lambda_value'])*_curvature_penalty(x),sparse.csr_matrix((len(acqs),len(acqs)))))
 D=sparse.hstack([sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)),sparse.csr_matrix((n-1,len(acqs)))])
 gauge=sparse.csr_matrix((np.ones(len(acqs))/len(acqs),(np.zeros(len(acqs)),n+np.arange(len(acqs)))),shape=(1,nv));A=sparse.vstack([D,gauge]);lower=np.r_[float(audit['minimum_slope_m_per_m'])*np.diff(x),0.];upper=np.r_[np.full(n-1,np.inf),0.]
 offsets=o.groupby('acquisition_id').acquisition_stage_offset_m.first().reindex(acqs).to_numpy();mean=offsets.mean();initial=np.r_[f.fitted_wse_m-center+mean,offsets-mean]
 for mode in ['equal_observation','equal_cycle']:
  weights=np.ones(N) if mode=='equal_observation' else N/(o.cycle.nunique()*o.groupby('cycle').wse_m.transform('size').to_numpy())
  P=2*mat.T@sparse.diags(weights)@mat+penalty;q=-2*mat.T@(weights*y)
  fit,qa=qp(P,q,A,lower,upper,initial,tol=1e-6,maxiter=120000)
  residual=y-mat@fit
  controls.append(dict(example=ex,mode=mode,observation_rms_m=float(np.sqrt(np.mean(residual**2))),iterations=qa['iterations'],max_profile_change_m=float(abs(fit[:n]+center-f.fitted_wse_m).max()),primal_residual=qa['primal_residual'],dual_residual=qa['dual_residual']))
  pd.DataFrame({'reference_section_dist_out_m':x,'fitted_wse_m':fit[:n]+center}).to_parquet(out/f'control_{mode}_profile.parquet',index=False)
 print(ex,'verified and controls solved',flush=True)
pd.DataFrame(checks).to_csv(OUT/'validation.csv',index=False);pd.DataFrame(controls).to_csv(OUT/'profile_controls.csv',index=False)
