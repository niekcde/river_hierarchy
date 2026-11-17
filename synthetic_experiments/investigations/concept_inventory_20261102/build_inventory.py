"""Exhaustive concept-level taxonomy of the frozen 205 candidate columns."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path('/Volumes/SyntheticResults20261027/analysis/broad_descriptors_20261030/geometry_with_manning/candidate_catalog.csv')
OUT=ROOT/'synthetic_experiments/notes/geometry_concept_inventory_20261102.json'
# Each tuple is one interpretive middle-level concept, not an independence claim.
ROWS=[
('Allocated route length distribution','L B C','route_length_mean route_length_sd route_length_cv route_length_min route_length_max route_length_skew log1p__B_ratio_route_length_mean log1p__B_ratio_route_length_sd length_delta_km route_spread_change_m route_spread_change_over_parent_mean',r'L_r=\sum_{e\in r}L_e,\quad P(R=r)=\pi_r','Lᵣ = ∑ₑ∈ᵣ Lₑ; P(R = r) = πᵣ','Mean, SD, CV, minimum, maximum and skewness, plus parent–child comparisons and extra scalings. Route probabilities come from width splitting. Minimum and maximum themselves do not use those weights.'),
('Equally weighted route length distribution','L C','path_length_cv path_length_range_norm',r'P_0(R=r)=1/N','P₀(R = r) = 1/N','Unweighted path-length CV and range divided by equivalent length. The CV uses equal path weights; the range normalisation additionally introduces width through L_eq.'),
('Reach width distribution','L B','width_mean width_sd width_cv width_harmonic width_p10 width_p90 log1p__B_ratio_width_cv',r'(B_e,w_e),\quad w_e=L_e/\sum_kL_k','(Bₑ, wₑ); wₑ = Lₑ / ∑ₖ Lₖ','Length-weighted mean, SD, CV, harmonic mean and P10/P90 widths. The transformed width-CV ratio is included here.'),
('Reach slope distribution','L S; B C for inverse slope','slope_mean slope_min slope_max slope_cv parent_slope_slow',r'(S_e,w_e),\quad w_e=L_e/\sum_kL_k','(Sₑ, wₑ); wₑ = Lₑ / ∑ₖ Lₖ','Mean, extremes and CV use the saved geometric slopes. The additional inverse-slope summary uses allocated-length weights: ∑ₑ fₑLₑSₑ⁻¹ᐟ² / ∑ₑ fₑLₑ.'),
('Total channel length','L','total_length',r'L_{\rm tot}=\sum_eL_e','L_tot = ∑ₑ Lₑ','Sum of unique physical reach lengths. Parallel channels contribute separately; this is not expected route length.'),
('Channel area','L B','area_proxy',r'A=\sum_eL_eB_e','A = ∑ₑ LₑBₑ','Channel-area proxy, not polygon-union area and not weighted by discharge allocation.'),
('Affected domain extent','L B C U','domain_size',r'D_U=10^{-3}\sum_{e\in U}f_eL_e','Dᵤ = 10⁻³ ∑ₑ∈ᵁ fₑLₑ','Allocation-weighted selected source-unit length in kilometres. U is the selected membership, not a fifth independent geometric ingredient.'),
('Local bifurcation allocation distribution','B C','dominance_mean dominance_sd effective_branches_mean effective_branches_sd split_entropy_mean split_entropy_sd',r'a_{vj}=B_{vj}/\sum_kB_{vk}','aᵥⱼ = Bᵥⱼ / ∑ₖ Bᵥₖ','At each split: dominance maxⱼ aᵥⱼ, effective branches 1/∑ⱼaᵥⱼ², and normalised entropy −∑ⱼaᵥⱼln(aᵥⱼ)/ln(kᵥ). Visit-weighted means and SDs summarise these across bifurcations.'),
('Bifurcation exposure distribution','B C','split_exposure_mean split_exposure_sd expected_splits',r'p_v=\sum_{e\to v}f_e,\quad E[N_{\rm split}]=\sum_{v\in\mathcal B}p_v','pᵥ = ∑ₑ→ᵥ fₑ; E[N_split] = ∑ᵥ∈ℬ pᵥ','The inlet has p_in = 1; other node probabilities sum incoming fractions. Unweighted mean and SD over bifurcations, plus their sum, give exposure summaries and expected splits.'),
('Exposure weighted splitting intensity','B C','split_impact_mean split_impact_sd split_impact_sum',r'I_v=p_v\left(1-\sum_ja_{vj}^2\right)','Iᵥ = pᵥ (1 − ∑ⱼ aᵥⱼ²)','Mean, SD and sum of local splitting intensity. The mean/SD are unweighted across bifurcations because exposure is already in Iᵥ.'),
('Allocated route diversity','B C','route_entropy log1p__B_ratio_route_entropy route_entropy_change',r'H_\pi=-\sum_r\pi_r\ln\pi_r','Hπ = −∑ᵣ πᵣ ln(πᵣ)','Entropy of complete-route allocation. Equals accumulated visit-weighted local entropy; distinct from mean normalised local entropy.'),
('Branch distance contrasts','L B C','branch_length_contrast_mean branch_length_contrast_sd branch_length_variance_sum',r'd_{vj}=L_{vj}+\mu_{L,j},\quad c_v^2=\sum_ja_{vj}(d_{vj}-\bar d_v)^2','dᵥⱼ = Lᵥⱼ + μL,ⱼ; cᵥ² = ∑ⱼ aᵥⱼ(dᵥⱼ − d̄ᵥ)²','Here μL,ⱼ is expected remaining distance from downstream node j and d̄ᵥ = ∑ⱼaᵥⱼdᵥⱼ. Visit-weighted mean and SD of cᵥ and the variance sum ∑ᵥpᵥcᵥ². The latter equals route-length variance, so this concept overlaps mathematically with allocated route spread.'),
('Equivalent path width distribution','L B C','equivalent_width dominant_width_fraction effective_n_paths_width width_entropy width_evenness',r'W_r={L_r\over\sum_{e\in r}L_e/B_e},\quad q_r={W_r\over\sum_sW_s}','Wᵣ = Lᵣ / ∑ₑ∈ᵣ(Lₑ/Bₑ); qᵣ = Wᵣ / ∑ₛ Wₛ','Equivalent total width ∑ᵣWᵣ, dominance max qᵣ, entropy Hq, evenness Hq/ln N and effective count exp(Hq). qᵣ is not routed allocation πᵣ. Evenness is conventionally 1 for N = 1.'),
('Equivalent path length','L B C','equivalent_length',r'L_{eq}=\sum_rq_rL_r','L_eq = ∑ᵣ qᵣLᵣ','Complete-path length weighted by harmonic path width. Separated from allocated route length because qᵣ and πᵣ differ.'),
('Length width association','L B C','signed_length_width_association',r'A_{LW}={L_{eq}\over N^{-1}\sum_rL_r}-1','A_LW = L_eq / [(1/N) ∑ᵣ Lᵣ] − 1','Positive when harmonic path width gives relatively more weight to longer routes; zero when all route lengths are equal.'),
('Network elongation','L B C','area_elongation elongation',r'E_A=\mu_L^2/A,\quad E_{eq}=L_{eq}/W_{eq}','E_A = μL²/A; E_eq = L_eq/W_eq','Two distinct length-versus-width shape definitions. Grouped as one geometric concept, not asserted to be equivalent.'),
('Route elongation relative to shortest route','L B C','sinuosity_proxy',r'\tau=\mu_L/L_{\min}','τ = μL / L_min','Allocated mean route length relative to the shortest directed route. Not map-based channel sinuosity.'),
('Topological architecture','C','bifurcations cycle_rank junction_excess log1p_paths n_valid_paths topologic_complexity_score',r'\mathcal T=(N_b,\ E-V+C_w,\ \sum_v(k_v-1)_+,\ N)','T = (N_b, E − V + C_w, ∑ᵥ max(kᵥ − 1, 0), N)','E, V and C_w count edges, vertices and weak components; kᵥ is outgoing degree. Includes branch count, undirected cycle rank, junction excess, path count and transformations. Historical complexity = ln(1+N)+ln(1+i_b)+ln(1+i_c), where internal bifurcations/confluences exclude the boundary source/sink.'),
('Branching density','L C; B for equivalent length','bifurcation_density branching_density_by_length',r'\rho_b=N_b/L_{tot},\quad \rho_{bc}=(i_b+i_c)/L_{eq}','ρ_b = N_b/L_tot; ρ_bc = (i_b + i_c)/L_eq','Two normalisations of branching architecture by length; the equivalent-length version also depends on width.'),
('Domain head loss','L S C','head_drop',r'\Delta H=z_{in}-z_{out}=\sum_{e\in r}S_eL_e','ΔH = z_in − z_out = ∑ₑ∈ᵣ SₑLₑ','Head drop over the domain. Path-sum form assumes the common consistent node-head field used in the synthetic construction.'),
('Manning geometric accumulation','L B S C','delta_G_manning',r'G_M=\sum_eL_eB_e^{2/5}f_e^{3/5}S_e^{-3/10}','G_M = ∑ₑ Lₑ Bₑ²ᐟ⁵ fₑ³ᐟ⁵ Sₑ⁻³ᐟ¹⁰','Only child-minus-parent ΔG_M is in this 205-column inventory. Sums cover whole networks, with allocation recomputed per state; no depth, n or simulated discharge is an input.')
]
def main():
 catalog=list(csv.DictReader(SOURCE.open()));assert len(catalog)==205
 assignment={};concepts=[]
 for i,(name,ingredients,metrics,tex,equation,note) in enumerate(ROWS,1):
  metrics=metrics.split()
  for metric in metrics:assert metric not in assignment;assignment[metric]=i
  members=[r for r in catalog if r['metric'] in metrics]
  assert set(metrics)=={r['metric'] for r in members},(name,set(metrics)-{r['metric'] for r in members})
  concepts.append(dict(id=i,name=name,ingredients=ingredients,count=len(members),metrics=[dict(metric=m,count=sum(r['metric']==m for r in members)) for m in metrics],equation=equation,latex=tex,note=note,candidates=[r['variable'] for r in members]))
 assert set(assignment)=={r['metric'] for r in catalog}
 assert sum(r['count'] for r in concepts)==205
 assert len({c for r in concepts for c in r['candidates']})==205
 payload=dict(reference='SYN-GEOMETRIC-CONCEPT-INVENTORY-2026-11-02',date='2026-11-02',n_concepts=len(concepts),n_candidates=205,n_metric_labels=len(assignment),grouping='Interpretive taxonomy by geometric object and weighting; exhaustive non-overlapping assignment of candidate columns, not statistical independence or a new selection.',source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),concepts=concepts)
 OUT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
 for r in concepts:print(f"{r['id']:2} {r['count']:2} {r['name']}")
 print('TOTAL',sum(r['count'] for r in concepts))
if __name__=='__main__':main()
