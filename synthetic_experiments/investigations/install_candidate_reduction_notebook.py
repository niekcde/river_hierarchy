"""Install the completed pre-RF candidate-reduction investigation."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[2]
TAG='candidate-reduction-20261030'
def install():
 p=ROOT/'synthetic_experiments/notebooks/synthetic_predictor_validation.ipynb';n=nbformat.read(p,4)
 n.cells=[c for c in n.cells if TAG not in c.metadata.get('tags',[])]
 cells=[nbformat.v4.new_markdown_cell('''## 10. Reducing 205 representations before RF testing

**SYN-CANDIDATE-REDUCTION-2026-10-30.** This is a completed mathematical, redundancy and mechanism audit—not an RF subset search. Read the [full scientific report](../notes/2026-10-30_geometry_candidate_reduction.md) for definitions, decisions, reproducibility and draft methods/results text.

**Outcome:** 139 complete representations vary within at least one type; 22 vary only between types; 40 are partly defined; four are invariant across the grid. Twenty-seven complete representations are empirical affine duplicates. Conservative correlation clustering leaves 69–114 groups depending on threshold, so correlation alone does not establish a small final set.

**Proposed pool:** 14 primary candidates, three replacement candidates, two topology-context reserves. This is an explicit analyst-curated pool for subsequent grouped prediction tests, not 19 proven independent predictors or a recommendation to fit all 19 together.

**Numerical refinement of Section 8:** full-column scaling removes subtraction roundoff from theoretically zero changes. Finite ratios/skewness with roundoff-only denominators are treated as undefined in audit copies. Source experiment outputs remain unchanged. All missing entries have an identified mathematical reason; no imputation is used.'''),
 nbformat.v4.new_code_cell('''REDUCTION=Path('/Volumes/SyntheticResults20261027/analysis/candidate_reduction_20261030')
for name in ['provenance.json','shortlist_provenance.json']:
    provenance=json.loads((REDUCTION/name).read_text())
    for path,digest in provenance['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
candidate_inventory=pd.read_csv(REDUCTION/'inventory.csv')
candidate_decisions=pd.read_csv(REDUCTION/'candidate_decisions.csv')
candidate_shortlist=pd.read_csv(REDUCTION/'shortlist.csv')
assert len(candidate_inventory)==len(candidate_decisions)==205 and len(candidate_shortlist)==19
assert candidate_shortlist.minimum_type_coverage.eq(100).all()
display(Image(filename=str(REDUCTION/'01_inventory.png')))
display(candidate_decisions.groupby('decision').size().rename('representations').to_frame())
print('Full ledger:',REDUCTION/'candidate_decisions.csv')
'''),nbformat.v4.new_markdown_cell('''### 10a. Which variables are proposed, and why?

**Primary** means worth taking into the small-pool prediction stage; it does not mean independently useful. **Alternative** means try a replacement before automatically adding another correlated representation. **Context reserve** means this grid cannot establish its within-type value, but real topology may need it.

Weak marginal correlates such as width CV, dominance and entropy are retained on mechanism grounds, not because the response screen proved an interaction. They must earn their place in later predictive testing. The complete 205-entry ledger also preserves all deferred candidates.'''),
 nbformat.v4.new_code_cell('''display(candidate_shortlist[['label','tier','mechanism','reason']])
display(Image(filename=str(REDUCTION/'02_response_associations.png')))
'''),nbformat.v4.new_markdown_cell('''### 10b. Same information, or merely similar correlation?

Clustering uses predictor–predictor correlations, not similarity in response correlations. Strong pooled and within-type correlations must agree, and the variables must vary in the same types. Spearman measures monotonic strength; Pearson tests linearity. Neither is a physical effect-size coefficient.

Manning ΔG and mean route-length change have pooled Spearman 0.989 but Pearson 0.929: highly related, not identical. Equivalent-length change has Pearson ≈0.999 with mean route-length change but uses different weights and can change in the near-null main-island case. Parent route spread and spread change have perfect absolute within-type correlation in six types but only −0.857 pooled correlation; the type-specific mappings differ.

The correlation figure below is the pooled view. Exported pairwise tables contain the conservative within-type evidence used for clustering.'''),
 nbformat.v4.new_code_cell('''candidate_clusters=pd.read_csv(REDUCTION/'clusters.csv')
display(candidate_clusters.groupby('threshold').cluster.nunique().rename('groups').to_frame())
display(Image(filename=str(REDUCTION/'03_predictor_correlation.png')))
display(Image(filename=str(REDUCTION/'05_pair_examples.png')))
'''),nbformat.v4.new_markdown_cell('''### 10c. Mechanisms and robustness across hydraulic approaches

A highly correlated predictor can still respond differently to width or slope changes. Main-effect shares below quantify how controls vary the descriptors; they are not predictor importance or added predictive value. Interactions are exported in the source factorial tables.

The method comparison uses the same six collapse types. Fixed-Manning and variable-Manning rankings agree closely; BASED rankings differ. This is evidence that hydraulic closure conditions the geometry–response relationship, not proof that one closure is universally correct.

The full report includes omission of each of 24 factor/level combinations, omission of each type, numerical-tolerance checks, and a tie diagnostic. Centroid rounding to one second increases Manning ΔG Spearman from ≈0.981 to ≈1 because tiny differences in theoretically tied zero-change cases otherwise receive separate ranks. Primary results remain unrounded; this diagnostic is not an error bound.'''),
 nbformat.v4.new_code_cell('''display(Image(filename=str(REDUCTION/'04_mechanisms.png')))
method_screen=pd.read_csv(REDUCTION/'common_support_methods.csv')
show_variables=['delta_G_manning','B_delta_route_length_mean','B_delta_area_proxy','B_parent_route_length_mean','domain_size']
display(method_screen[method_screen.variable.isin(show_variables)&method_screen.response.eq('centroid_shift_minutes')].pivot(index='variable',columns='method',values='mean_abs_spearman').round(3))
level_screen=pd.read_csv(REDUCTION/'leave_level_out_summary.csv')
display(level_screen[level_screen.response.eq('centroid_shift_minutes')&level_screen.n_types.eq(6)].groupby('variable').mean_abs_spearman.agg(['min','max']).round(3))
print('These ranges are design sensitivities, not confidence intervals or held-out performance.')
'''),nbformat.v4.new_markdown_cell('''### 10d. Scientific conclusion and next boundary

The main synthetic evidence supports distance-change and route-heterogeneity descriptors, with a Manning-specific combined descriptor providing a strong centroid axis. Peak shift is less well represented by the same single axis. Width, slope and allocation remain candidate modifiers; topology is under-tested within fixed synthetic types.

No RF fits were performed for this reduction. The next step is a grouped predictive comparison of small subsets and replacement variants. That later step must evaluate whether weak marginal candidates add information, whether correlated representatives can be exchanged, and whether the retained topology context transfers to real cases. The present shortlist is exploratory and must not be described as an optimal or independently validated final predictor set.''')]
 for c in cells:c.metadata['tags']=[TAG]
 n.cells.extend(cells);nbformat.write(n,p)
if __name__=='__main__':install()
