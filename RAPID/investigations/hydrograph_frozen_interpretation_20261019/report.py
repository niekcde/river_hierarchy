"""Build self-contained figures and a dedicated results notebook from verified folds."""
from pathlib import Path
import base64,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
from run import OUT, ROOT, C, LABEL, LABELS, VIEWS, raw
FIG=OUT/'figures'

def save(fig,name):
 fig.savefig(FIG/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)

def ale_stability():
 local=pd.read_parquet(OUT/'ale_local_differences.parquet')
 rows=[]
 for (view,ex,c,bins),g in local[local.grid=='network'].groupby(['view','example','feature','bins']):
  g=g.sort_values('bin')
  if g.total.max()==0:continue
  # Do not bridge empty bins in exclusion views.
  if (g.n==0).any():continue
  delta=g.local_difference_minutes
  effect=delta.cumsum()-.5*delta
  rows.append(dict(view=view,example=ex,feature=c,bins=bins,endpoint_contrast_minutes=float(effect.iloc[-1]-effect.iloc[0]),positive_local_bins=int((delta>1e-8).sum()),negative_local_bins=int((delta< -1e-8).sum()),flat_local_bins=int((abs(delta)<=1e-8).sum()),min_rows_per_bin=int(g.n.min()),total_rows=int(g.n.sum())))
 t=pd.DataFrame(rows);t.to_csv(OUT/'ale_direction_by_network.csv',index=False)
 t.groupby(['view','feature','bins']).agg(mean_endpoint_contrast_minutes=('endpoint_contrast_minutes','mean'),min_endpoint_contrast=('endpoint_contrast_minutes','min'),max_endpoint_contrast=('endpoint_contrast_minutes','max'),positive_networks=('endpoint_contrast_minutes',lambda x:int((x>1e-8).sum())),negative_networks=('endpoint_contrast_minutes',lambda x:int((x< -1e-8).sum())),valid_networks=('example','count')).to_csv(OUT/'ale_direction_summary.csv')


def figures():
 FIG.mkdir(exist_ok=True)
 s=pd.read_parquet(OUT/'shap_values.parquet');a=pd.read_csv(OUT/'ale_summary.csv');local=pd.read_parquet(OUT/'ale_local_differences.parquet')
 # Signed contributions, all networks individually visible; summaries use equal weights.
 fig,axs=plt.subplots(4,2,figsize=(13,16));nets=sorted(s.example.unique());colors=plt.cm.tab20(np.linspace(0,1,len(nets)))
 for ax,c in zip(axs.flat,C):
  for ex,col in zip(nets,colors):
   g=s[(s.feature==c)&(s.example==ex)];ax.scatter(g.value,g.shap_minutes,s=6,alpha=.45,color=col,label=ex)
  ax.axhline(0,color='gray',lw=.6);ax.set(xlabel=LABEL[c],ylabel='Signed SHAP (minutes)')
  if c=='domain_size':ax.set_xscale('log')
 axs.flat[-1].axis('off');axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),ncol=3,title='Held network',loc='center')
 fig.suptitle('Frozen seven-input forest: held-network SHAP\nEach point is a comparison; density is not equal-network weighting',fontsize=14);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'signed_shap')
 fig,axs=plt.subplots(4,2,figsize=(13,16))
 for ax,c in zip(axs.flat,C):
  ll=local[(local.feature==c)&(local.view=='all')&(local.bins==8)&(local.grid=='network')]
  for ex,g in ll.groupby('example'):
   g=g.sort_values('bin');delta=g.local_difference_minutes
   val=delta.cumsum()-.5*delta;val-=np.average(val,weights=g.n)
   ax.plot(raw(c,(g.lower+g.upper)/2),val,color='gray',alpha=.35,lw=.7)
  for bins,col in [(4,'#2a9d8f'),(8,'#264653'),(12,'#e76f51')]:
   g=a[(a.feature==c)&(a.view=='all')&(a.bins==bins)].sort_values('bin');x=raw(c,(g.lower+g.upper)/2)
   ax.plot(x,g.ale_minutes,'o-',color=col,lw=1.4,ms=3,label=f'{bins} bins, equal-network')
   if bins==8:
    for xx,yy,n in zip(x,g.ale_minutes,g.networks):ax.annotate(str(n),(xx,yy),xytext=(0,5),textcoords='offset points',fontsize=7)
  ax.axhline(0,color='gray',lw=.6);ax.set(xlabel=LABEL[c],ylabel='Centered ALE (minutes)')
  if c=='domain_size':ax.set_xscale('log')
 axs.flat[-1].axis('off');axs.flat[-1].legend(*axs.flat[0].get_legend_handles_labels(),loc='center')
 fig.suptitle('First-order ALE: gray = network curves; labels = contributing networks\nCommon bins mix fold-specific local effects; lines are descriptive, not confidence intervals',fontsize=13);fig.tight_layout(rect=[0,0,1,.96]);save(fig,'ale')
 fig,axs=plt.subplots(4,2,figsize=(13,14))
 for ax,c in zip(axs.flat,C):
  g=a[(a.feature==c)&(a.view=='all')&(a.bins==8)].sort_values('bin')
  ax.bar(np.arange(len(g)),g.rows,color='#90c4cc');ax.set(xlabel=LABEL[c]+' (bin center)',ylabel='Comparisons')
  ax.set_xticks(np.arange(len(g)),[f'{v:.2g}' for v in raw(c,(g.lower+g.upper)/2)],rotation=40)
  for i,r in enumerate(g.itertuples()):ax.text(i,r.rows,f'{r.networks} nets',ha='center',va='bottom',fontsize=7)
 axs.flat[-1].axis('off');fig.suptitle('ALE support: counts do not imply independent samples');fig.tight_layout(rect=[0,0,1,.97]);save(fig,'ale_support')
 # Interaction allocations plus joint occupancy, using equal-network weights.
 ii=pd.read_parquet(OUT/'shap_width_size_interaction.parquet');x=raw(C[3],ii[C[3]]);y=ii[C[1]].to_numpy()
 fig,axs=plt.subplots(1,3,figsize=(16,4.8));sc=axs[0].scatter(x,ii.interaction_minutes,c=np.log1p(y),cmap='viridis',s=10,alpha=.6);fig.colorbar(sc,ax=axs[0],label='ln(1 + parent size / km)');axs[0].set(xlabel='Mean width ratio',ylabel='Pair SHAP contribution (minutes)');axs[0].axhline(0,color='gray',lw=.6)
 # quantile occupancy table in transformed width/size coordinates
 from run import E,quantiles
 w=E.weights(ii);xe=np.unique(quantiles(ii[C[3]],w,np.linspace(0,1,6)));ye=np.unique(quantiles(y,w,np.linspace(0,1,6)))
 ix=np.clip(np.searchsorted(xe,ii[C[3]],side='right')-1,0,len(xe)-2);iy=np.clip(np.searchsorted(ye,y,side='right')-1,0,len(ye)-2)
 ii=ii.assign(ix=ix,iy=iy);rows=[];val=np.full((len(ye)-1,len(xe)-1),np.nan);nn=np.zeros_like(val)
 for (j,k),g in ii.groupby(['iy','ix']):
  b=g.groupby('example').interaction_minutes.mean();val[j,k]=b.mean();nn[j,k]=len(b);rows.append(dict(width_bin=k,size_bin=j,rows=len(g),networks=len(b),mean_pair_minutes=b.mean(),width_lower=np.expm1(xe[k]),width_upper=np.expm1(xe[k+1]),size_lower=ye[j],size_upper=ye[j+1]))
 pd.DataFrame(rows).to_csv(OUT/'width_size_joint_support.csv',index=False)
 vmax=np.nanmax(abs(val));im=axs[1].imshow(val,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax);fig.colorbar(im,ax=axs[1],label='Equal-network pair SHAP (min)')
 axs[2].imshow(nn,origin='lower',cmap='Blues',vmin=0,vmax=15)
 for ax in axs[1:]:
  ax.set(xlabel='Width-ratio quantile bin',ylabel='Parent-size quantile bin');ax.set_xticks(range(len(xe)-1));ax.set_yticks(range(len(ye)-1))
 for j in range(nn.shape[0]):
  for k in range(nn.shape[1]):axs[2].text(k,j,str(int(nn[j,k])),ha='center',va='center')
 axs[2].set_title('Contributing networks (empty = unsupported)');fig.tight_layout();save(fig,'width_size_interaction')
 # Cross-network signs and importance variability.
 sh=pd.read_csv(OUT/'shap_by_network.csv',dtype={'example':str});cr=pd.read_csv(OUT/'response_correlations.csv',dtype={'example':str});pm=pd.read_csv(OUT/'permutation_by_network_all.csv',dtype={'example':str})
 fig,axs=plt.subplots(1,3,figsize=(17,6))
 tables=[sh[sh.view=='all'].pivot(index='example',columns='feature',values='signed_shap_value_rho')[C],cr[(cr.view=='all')&(cr.example!='pooled_rows')].pivot(index='example',columns='feature',values='rho')[C],pm[pm.group.isin(C)].pivot(index='example',columns='group',values='delta_relative_pp')[C]]
 for k,(ax,t) in enumerate(zip(axs,tables)):
  lim=1 if k<2 else np.nanmax(abs(t.to_numpy()));im=ax.imshow(t,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');fig.colorbar(im,ax=ax,shrink=.7);ax.set_xticks(range(7),LABELS,rotation=60,ha='right');ax.set_yticks(range(len(t)),t.index)
  ax.set_title(['Value vs signed SHAP (rho)','Value vs response (rho)','Permutation relative-loss increase (pp)'][k])
 fig.tight_layout();save(fig,'network_stability')
 pc=pd.read_csv(OUT/'predictor_correlations.csv',dtype={'example':str});fig,axs=plt.subplots(1,2,figsize=(12,6))
 for ax,typ in zip(axs,['pooled_rows','mean_within']):
  sub=pc[(pc.view=='all')&(pc.example=='pooled_rows')] if typ=='pooled_rows' else pc[(pc.view=='all')&(pc.example!='pooled_rows')].groupby(['feature1','feature2'],as_index=False).rho.mean()
  mat=np.eye(7)
  for r in sub.itertuples():j=C.index(r.feature1);k=C.index(r.feature2);mat[j,k]=mat[k,j]=r.rho
  im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(7),LABELS,rotation=60,ha='right');ax.set_yticks(range(7),LABELS);ax.set_title(typ.replace('_',' '))
  for j in range(7):
   for k in range(7):ax.text(k,j,f'{mat[j,k]:.2f}',ha='center',va='center',fontsize=7)
 fig.tight_layout();save(fig,'predictor_correlations')


def notebook():
 n=nb.v4.new_notebook();cells=[]
 def md(t):cells.append(nb.v4.new_markdown_cell(t))
 def code(t):cells.append(nb.v4.new_code_cell(t))
 def picture(name):
  c=nb.v4.new_markdown_cell(f'![{name}](attachment:{name}.png)');c.attachments={f'{name}.png':{'image/png':base64.b64encode((FIG/f'{name}.png').read_bytes()).decode()}};cells.append(c)
 def table(file,query=None,cols=None):
  x=pd.read_csv(OUT/file)
  if query:x=x.query(query)
  if cols:x=x[cols]
  cell=nb.v4.new_code_cell(f"table = pd.read_csv(OUT / {file!r})\n"+(f"table = table.query({query!r})\n" if query else '')+(f"table = table[{cols!r}]\n" if cols else '')+'display(table)')
  cell.execution_count=len([c for c in cells if c.cell_type=='code'])+1
  cell.outputs=[nb.v4.new_output('display_data',data={'text/html':x.to_html(index=False,float_format=lambda v:f'{v:.4g}'),'text/plain':x.to_string(index=False)})];cells.append(cell)
 md('# Frozen seven-variable forest: prediction importance and cross-network physical consistency\n\nDedicated analysis, 19 October 2026. **These results reconstruct the final tuned seven-input forests**, not the historical relative-sweep forests. The baseline and historical output directory are unchanged. Only the 2,170 baseline partial-collapse comparisons across 15 networks are included. Whole-network single-centerline cases remain a separate transfer evaluation.\n\n'+(Path(__file__).with_name('findings.txt')).read_text())
 code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display\nROOT = Path.cwd()\nwhile not (ROOT / 'RAPID').exists():\n    if ROOT == ROOT.parent: raise RuntimeError('Run inside river-hierarchy')\n    ROOT = ROOT.parent\nOUT = ROOT / 'RAPID/outputs/hydrograph_frozen_interpretation_20261019'")
 md('## Provenance and verification\n\nExact settings, seven-column order, seed 20261014, training-network weights, fold-fitted preprocessing and target scaling are inherited from `drop_exposure_check.py`. SHA256 hashes and package versions are in `design.json`. The baseline prediction table is filtered to `seven_inputs` and joined one-to-one by network/state. Each network is excluded completely from its training fold. Settings were previously selected using these networks; this is exploratory fixed-setting interpretation, not fresh nested validation.\n\nSHAP uses exact tree-path-dependent TreeSHAP, explaining the held-network prediction relative to each forest’s own weighted training-path baseline. Additivity of both SHAP and the interaction decomposition is checked numerically in minutes. Fold baselines differ. The 2,170 comparisons are **not 2,170 independent network samples**; all aggregate importance summaries give networks equal weight.')
 table('verification.csv',cols=['example','prediction_max_error_minutes','shap_additivity_max_error_minutes','interaction_additivity_max_error_minutes'])
 md('## Held-network permutation importance\n\nTwenty repeats per network, identical shuffle orders across groups for paired comparison. Group members share one row permutation. Relative loss is 100 × |prediction−response| / max(|response|, 1 minute). Positive changes mean disrupted information helped prediction. Squared-error changes are in minutes². R² decrease divides the equal-network MSE increase by the fixed equal-network response variance for that view; it is not a mean of network R² values.\n\nGroups: length geometry = mean-length ratio, parent size, length-SD ratio; width geometry = mean-width and width-CV ratios; branching = entropy ratio and parent dominance; width–entropy = both width ratios and entropy ratio; length–width = all length and width variables. Joint shuffling preserves dependence within a group but breaks its dependence with other inputs. Importance is not additive, and individual permutations may generate physically implausible combinations. Within-network permutation cannot evaluate network-constant information.\n\nNetwork SD/min/max describe heterogeneity, not confidence intervals. Shuffle Monte Carlo SD describes finite-repeat randomness, not uncertainty across populations of networks.')
 table('permutation_summary.csv',"view == 'all'",['group','delta_relative_pp','delta_relative_pp_sd','delta_relative_pp_min','delta_relative_pp_max','delta_relative_pp_positive_networks','delta_mse_minutes2','r2_decrease','shuffle_mc_sd_relative_pp'])
 md('## SHAP magnitude and cross-network stability\n\nMean absolute SHAP measures allocation of a prediction, not predictive gain, direction, causality or statistical significance. Correlated predictors can share allocations. Signed SHAP/value rank correlations summarize a dependence cloud, not a controlled derivative. Inspect ALE alongside them. The rank range and network magnitude ranges show stability without treating comparisons as independent replicates.')
 table('shap_summary.csv',"view == 'all'")
 picture('network_stability');picture('signed_shap')
 md('## Accumulated local effects and support\n\nFirst-order ALE changes one transformed predictor between adjacent local bin boundaries, predicts each held-network row using its held-out forest, and averages local changes. Common boundaries are quantiles of the equal-network mixture of baseline inputs. Each supported network receives equal weight per bin; unsupported networks are omitted and explicitly counted. The accumulated curve is centered using equal-network bin frequencies. Thus this describes a collection of cross-fitted models, not one universal forest response function. Mid-bin values use half the bin’s local difference, the usual piecewise-linear display convention.\n\nGray curves use each network’s own eight quantile bins and centering; colored curves use 4, 8 and 12 common bins. Lines connect bin summaries, not mechanistic trajectories. Large bins and sparse joint support can still involve unrealistic perturbations; ALE reduces, but cannot remove, that problem. Feature axes show original ratios, while perturbations use the model’s ln(1+ratio) coordinates. Point annotations count contributing networks. `ale_local_differences.parquet`, `ale_summary.csv` and `support.csv` preserve exact boundaries, row counts, network counts, local-difference variability and held values beyond training ranges.')
 picture('ale');picture('ale_support')
 table('ale_direction_summary.csv',"view == 'all' and bins == 8")
 md('Endpoint contrasts above compare the first and last supported mid-bin ALE values within each network’s own range; they are not a common physical contrast or proof of monotonicity. Bin-level positive/negative counts are saved in `ale_direction_by_network.csv`. Empty-bin exclusion curves are omitted from endpoint summaries rather than bridged.')
 md('## Width ratio × parent-size interaction\n\nPair values are twice the off-diagonal SHAP interaction entry, in minutes, preserving the full pair allocation. This is a decomposition of model predictions, not a physical interaction coefficient or a second-order ALE surface. The 5×5 joint table averages within each network then equally across contributing networks. Empty cells have no evidence; cells with few networks should not support broad physical claims. Exact ratio/size edges and counts are in `width_size_joint_support.csv`. A negative pair allocation means this pair pushes toward an advance relative to the SHAP decomposition, not that increasing width always advances the centroid.')
 table('interaction_summary.csv');picture('width_size_interaction')
 md('## Marginal correlations versus conditional model behavior\n\nPooled Spearman correlations use all rows and can be dominated by larger networks; they are shown deliberately as a comparison, not as the primary cross-network estimate. Primary summaries average within-network Spearman coefficients equally, with min/max and positive/negative counts. Near-constant variables (range < 1e−10), fewer than four observations or constant responses have undefined rho. No naive independent-row p-values are reported.\n\nDisagreement between marginal response correlation, SHAP/value correlation and ALE may reflect predictor redundancy, between-network composition, nonlinearities and interactions, or residual confounding. SHAP allocates predictions relative to a baseline, whereas ALE makes local conditional model perturbations. Neither identifies a causal effect. The positive direction of a centroid shift is delay; negative is advance. Geometry alone does not establish a monotonic hydraulic law for the full excess-volume centroid.')
 table('response_correlation_summary.csv',"view == 'all'");picture('predictor_correlations')
 md('## Summary exclusions: 54 and 37 / CB001_bubble_3\n\n**No sensitivity refits were performed.** Every view reuses the same reconstructed forests and predictions; the frozen predictor set and settings remain unchanged. Excluding 54 from summaries does not remove it from other networks’ training folds. Similarly, removing the extreme comparison from scoring does not remove its training influence on other held-network fits. Its own prediction already comes from a forest trained without all of network 37. Training influence requires a separately authorized refit study.\n\nPermutation exclusions retain the original full-network donor pool and remove rows only from score summaries; this isolates summary sensitivity without changing the perturbation experiment. SHAP and correlations filter summary rows. ALE retains common baseline bin edges and filters local differences, so remaining local-effect summaries can be compared on common intervals. The four views are all, without 54, without the extreme comparison, and without both; each view rebalances remaining networks equally and uses its own response variance for R².')
 table('scores_summary_exclusions.csv');table('shap_summary.csv',None,['view','feature','mean_abs_shap_minutes','direction_positive','direction_negative'])
 code("display(pd.read_csv(OUT/'permutation_summary.csv'))\ndisplay(pd.read_csv(OUT/'response_correlation_summary.csv'))\ndisplay(pd.read_parquet(OUT/'shap_values.parquet').query(\"example == '37' and state == 'CB001_bubble_3'\"))")
 md('## Reproduction and artifacts\n\nRun `run.py` then `report.py` in `RAPID/investigations/hydrograph_frozen_interpretation_20261019/`. The run requires the original NumPy 1.26.4 / sklearn 1.8.0 environment and SHAP 0.46.0. On this workstation:\n\n```sh\nMPLCONFIGDIR=/private/tmp/river_frozen_mpl /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/hydrograph_frozen_interpretation_20261019/run.py --shap-path /private/tmp/river-shap-deps --jobs 4 --repeats 20\nMPLCONFIGDIR=/private/tmp/river_frozen_mpl /opt/anaconda3/envs/river-hierarchy-rivgraph/bin/python RAPID/investigations/hydrograph_frozen_interpretation_20261019/report.py\n```\n\nVerified fold artifacts are resumable under an exact design fingerprint. Baseline files are never written. Results, full precision CSV/Parquet tables and PNG figures live in `RAPID/outputs/hydrograph_frozen_interpretation_20261019/`. This notebook embeds its figures and primary tables for review without rerunning the forests. Statistical companion models belong in a separate work area.')
 n.cells=cells;n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}};nb.validate(n);nb.write(n,ROOT/'RAPID/notebooks/hydrograph_frozen_forest_interpretation.ipynb')

if __name__=='__main__':
 (OUT/'findings.txt').write_text(Path(__file__).with_name('findings.txt').read_text())
 ale_stability();figures();notebook()
