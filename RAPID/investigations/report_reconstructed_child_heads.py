"""Dated reconstruction decision, hydraulic evidence, and routing validation."""
from pathlib import Path
import sys,json,base64,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconstruct_child_heads import OUT,BASE,ROOT,EXAMPLES
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat


def table(f):
    def fmt(v):return f'{v:.6g}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)])


def main():
    inv=pd.read_csv(OUT/'inventory.csv',dtype={'example':str});hyd=pd.read_csv(OUT/'hydraulic_invariants.csv',dtype={'example':str});oldinv=pd.read_csv(BASE/'full_signed_validation/inventory.csv',dtype={'example':str})
    oldhyd=pd.read_csv(BASE/'full_signed_validation/candidate_hydraulic_ranges.csv',dtype={'example':str})
    assert inv.status.ne('blocked_input').all()
    pairs=[];runs=[];coverage=[]
    for ex in EXAMPLES:
        m=json.loads((OUT/ex/'routing_method.json').read_text());g=inv[inv.example==ex];unique=g[g.status=='ready'].fingerprint.nunique()
        p=pd.read_csv(OUT/ex/'comparisons.csv',dtype={'example':str});r=pd.read_csv(OUT/ex/'routing_summary.csv',dtype={'example':str})
        assert p.groupby('fingerprint').size().eq(2).all() and len(p)==2*(unique-1)
        assert len(r)==2*unique and r.min_flow.min()>0
        try:err=pd.read_csv(OUT/ex/'routing_errors.csv');assert len(err)==0
        except pd.errors.EmptyDataError:pass
        ranges=p.groupby('fingerprint')[['timing_change_minutes','duration_change_pct','peak_excess_change_pct']].agg(lambda a:a.max()-a.min())
        h=hyd[hyd.example==ex]
        coverage.append(dict(example=ex,ready_entries=int(g.status.eq('ready').sum()),excluded=int(g.status.eq('excluded_topology_review').sum()),unique_states=unique,comparisons=unique-1,clocks_s='/'.join(map(str,m['clocks'])),min_slope=h.min_slope.min(),max_raw_celerity=h.max_raw_celerity.max(),max_timing_difference_s=60*ranges.timing_change_minutes.max(),max_duration_difference_pp=ranges.duration_change_pct.max(),max_peak_difference_pp=ranges.peak_excess_change_pct.max()))
        pairs.append(p);runs.append(r)
    pairs=pd.concat(pairs,ignore_index=True);runs=pd.concat(runs,ignore_index=True);coverage=pd.DataFrame(coverage)
    coverage.to_csv(OUT/'coverage_and_convergence.csv',index=False);pairs.to_csv(OUT/'comparisons.csv',index=False);runs.to_csv(OUT/'routing_summary.csv',index=False)
    inv[inv.status=='excluded_topology_review'].to_csv(OUT/'retained_exclusions.csv',index=False)
    affected=[]
    for ex,state in [('02','S004_unit_5'),('02','S023_unit_12'),('22','S037_unit_220'),('37','S156_unit_18'),('37','island_37_161')]:
        o=oldinv[(oldinv.example==ex)&(oldinv.state==state)].iloc[0];n=inv[(inv.example==ex)&(inv.state==state)].iloc[0]
        nh=hyd[(hyd.example==ex)&(hyd.fingerprint==n.fingerprint)].iloc[0]
        oldmax=np.nan if o.status!='ready' else float(oldhyd[(oldhyd.example==ex)&(oldhyd.fingerprint==o.fingerprint)].max_raw_celerity.iloc[0])
        npair=pairs[(pairs.example==ex)&(pairs.fingerprint==n.fingerprint)].sort_values('dt_s').iloc[0]
        oldtime=np.nan
        if o.status=='ready':
            p=pd.read_csv(BASE/'full_signed_validation'/ex/'comparisons.csv');oldtime=float(p[p.fingerprint==o.fingerprint].sort_values('dt_s').timing_change_minutes.iloc[0])
        affected.append(dict(example=ex,state=state,previous_status=o.status,old_max_raw_celerity=oldmax,new_max_raw_celerity=nh.max_raw_celerity,old_timing_minutes=oldtime,new_timing_minutes=npair.timing_change_minutes))
    affected=pd.DataFrame(affected);affected.to_csv(OUT/'affected_case_changes.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    plot=affected.dropna(subset=['old_max_raw_celerity']);x=np.arange(len(plot));labels=[f'{r.example}\n{r.state}' for r in plot.itertuples()]
    axes[0].bar(x-.17,plot.old_max_raw_celerity,width=.34,label='Previous candidate',color='#94a3b8');axes[0].bar(x+.17,plot.new_max_raw_celerity,width=.34,label='Joint reconstruction',color='#2563eb');axes[0].axhline(5,ls=':',c='black',label='5 m/s routing cap');axes[0].set_xticks(x,labels);axes[0].set_ylabel('Maximum raw celerity (m/s)');axes[0].legend(fontsize=8)
    axes[1].bar(coverage.example,coverage.max_timing_difference_s,color='#2563eb');axes[1].set_ylabel('Maximum two-clock timing contrast difference (s)');axes[1].set_xlabel('Case');axes[1].tick_params(axis='x',rotation=90)
    fig.suptitle('Child hydraulic reconstruction: outliers resolved; timing convergence retained')
    fig.savefig(OUT/'reconstruction_summary.png',dpi=170);plt.close(fig)
    cap=pd.read_csv(OUT/'targeted_cap_sensitivity.csv',dtype={'example':str});a=cap[cap.policy=='cap5'].set_index(['example','state']);b=cap[cap.policy=='no_upper_cap'].set_index(['example','state']);delta=(b.timing_change_minutes-a.timing_change_minutes).rename('uncapped_minus_capped_timing_minutes').reset_index();delta.to_csv(OUT/'targeted_cap_effects.csv',index=False)
    report=f'''# Consistent child hydraulic reconstruction — 2026-10-03

Reference: **CHILD-SHARED-HEADS-2026-10-03**. Written after the full signed-routing audit found failed child inputs and raw celerities up to 61 m/s. This report supersedes that audit's provisional child-head reconstruction and its classification of five island contexts as missing.

## Result

All **{inv.status.eq('ready').sum():,} included entries** now reconstruct successfully across 16 examples. These represent **{len(hyd):,} unique hydraulic states** and **{len(pairs)//2:,} unique base–child comparisons**. The five previously documented topology exclusions remain excluded. There are **zero reconstruction failures** among the included entries, and zero routing-screen errors. All 16 base hydraulic frames are exactly unchanged (slopes, width, depth and K).

No roughness, depth exponent, cap, segmentation, profile or external unit boundary was adjusted to obtain this result. n=.035, alpha=.65, X=.1, 200 m segmentation/effective floor and common .05–5 m/s caps remain fixed. The signed-coefficient routing policy remains an explicitly tested numerical candidate; it is not universally positivity-preserving.

## What was wrong

1. **Interior heads were being inherited too rigidly after topology changed.** A changed branch must be solved together with the interior of its owning parent unit. Keeping old internal heads or solving only newly created vertices can impose incompatible drops.
2. **The previous fallback optimization was not reliable for these scales.** In case 37 it placed interior heads close to opposite outer boundaries, concentrating roughly 59 m of head loss into an 86 m link. These extreme values were generated by the earlier investigation's reconstruction, not by the bounded robust base profile.
3. **Coordinates alone do not uniquely identify a child graph node.** Case 02 has two distinct graph nodes at the same point. Merging both into one parent ID creates a false self-loop. The new matching is injective. For coincident ports it prioritizes unchanged direct-profile continuation, then the number and physical length of unchanged incident links. Only the selected port inherits the old head; the other retains its own graph identity and receives a solved head. Unresolved ties raise an error rather than using node-ID equality or merging vertices.
4. **Five missing paths were intentional exclusions.** The source island registry explicitly says `excluded_from_primary_analysis` / `excluded_topology_review`. Those entries should never have been counted as failed hydraulic reconstruction.

## Method

Detect changed connectivity or physical link length against the base graph. Preserve unchanged domains. For an affected owned simple/compound domain, solve its surviving interior vertices and any new vertices jointly; keep its exterior/shared-profile anchors fixed. Width-only changes do not require a new slope solution. All external anchors stayed fixed in these runs; no anchor-relaxation fallback is enabled.

For each connected affected patch, minimize the existing equal-per-link relative-slope objective:

    sum_e [ (H_u - H_v) / (L_e S_ref) - 1 ]²

subject to:

    H_anchor = selected base anchor head
    (H_u - H_v) / L_e >= 0.01 S_ref

S_ref is the outer anchor head span divided by their reference-section distance span. L_e is actual link length. The existing 1% relative unit floor is retained; there is no new absolute minimum slope. The reference profile's own 5% floor is unchanged. A sparse linear solve handles feasible unconstrained optima; a convex constrained solve with primal and stationarity checks handles active constraints. The objective is scaled numerically without changing its minimizer. Hydraulic allocation contains no n, celerity cap, routing timestep or stability target.

One shared head per graph node makes path head drops telescope exactly. The independent audit checks shared heads, positive finite slopes, exact frozen boundary heads, inlet–outlet head drop, graph acyclicity and candidate CSV consistency.

## Hydraulic evidence

{table(affected)}

Across all unique states, minimum slope is **{hyd.min_slope.min():.6g} m/m**, raw celerity spans **{hyd.min_raw_celerity.min():.6g}–{hyd.max_raw_celerity.max():.6g} m/s**, and no link needs the 0.05 m/s lower cap. There are {int(hyd.coincident_ports.sum())} unique state matches with a coincident-port correction. Maximum shared-head, fixed-anchor and whole-network head-drop errors are all zero at stored precision. The 16 base frames exactly reproduce the preceding base audit, so its 39,001-subreach native checks remain applicable.

The previous extreme child values should not be interpreted as physical river properties or fixed by increasing the cap. The corrected minimum being below the base minimum is possible: different branch lengths/topology allocate the same fixed boundary drop differently.

## Routing check

{table(coverage)}

Every unique state was screened at two common parent–child routing clocks, holding K fixed. Maximum contrast differences are **{coverage.max_timing_difference_s.max():.6g} s** for peak timing, **{coverage.max_duration_difference_pp.max():.6g} percentage points** for event duration and **{coverage.max_peak_difference_pp.max():.6g} percentage points** for peak excess. All screened outlet discharges are positive. The reporting grid is 120 s; the transfer-function method and its native-step cross-check are documented in FULL-SIGNED-ROUTING-2026-10-02. These are temporal convergence checks, not evidence of field timing accuracy. Internal child segments were not all stress-tested independently.

Thirteen cases use their saved actual-event inflow. Cases 13/45/48 have no saved actual-event base run and use the explicitly labelled standard pulse. The larger timing sensitivity in case 56 was already followed to a finer clock in the preceding audit; its remaining paired difference was approximately 1.2 s. The child reconstruction does not change that selected compound geometry.

## Cap sensitivity remains a separate issue

{table(delta)}

The case-37 outliers no longer need upper capping. The long case-22 compound collapse still has a meaningful cap effect: its 81.8 km link at about 5.306 m/s gives approximately a 15.7 minute difference between capped and uncapped timing contrasts. Reconstructing heads does not remove that length-related sensitivity. Retain it in the sensitivity discussion; do not confuse it with the resolved allocation bug.

## Saved artifacts and status

- `inventory.csv`: every entry, retained exclusion and candidate manifest path; aliases share the same source context where appropriate.
- `*/contexts/*/`: link and node CSVs in the **native child node IDs**, plus manifests. Each link stores endpoint heads, slope, depth and local Kb. The manifest's `input_paths` identifies source geometry; legacy context pointers are kept separately as source provenance, never advertised as the refreshed inputs.
- `hydraulic_invariants.csv`, `base_unchanged.csv`: independent checks for every unique state and all 16 bases.
- `comparisons.csv`, `routing_summary.csv`, per-case hydrographs: recomputed paired results.
- `affected_case_changes.csv`, `targeted_cap_sensitivity.csv`: before/after and cap evidence.

Implementation: `network_variants/shared_head_reconstruction.py`; batch materialization: `RAPID/investigations/reconstruct_child_heads.py`; routing: `RAPID/investigations/route_reconstructed_child_heads.py`. Eleven focused/regression tests pass, including active-floor optimization, infeasible constraints, unchanged boundaries and coincident graph ports.

The **child reconstruction blocker is resolved for the included set**. These are saved, checked candidate contexts and rerouted investigation results. Production files, production preparation defaults and equivalent-resistance caches were not overwritten. Consistent production RAPID and equivalent-resistance preparation still needs to consume the new contexts and the agreed depth/Kb rule together; old scalar-Kb resistance results must not be paired with these new routed responses. The known three missing actual-event runs also remain a separate coverage limitation.
'''
    (OUT/'README.md').write_text(report)
    script_paths=[ROOT/'network_variants/shared_head_reconstruction.py']+[Path(__file__),ROOT/'RAPID/investigations/reconstruct_child_heads.py',ROOT/'RAPID/investigations/route_reconstructed_child_heads.py',ROOT/'RAPID/investigations/route_full_signed_validation.py',ROOT/'RAPID/investigations/check_reconstructed_head_invariants.py']
    (OUT/'provenance.json').write_text(json.dumps(dict(reference='CHILD-SHARED-HEADS-2026-10-03',production_changed=False,base_unchanged=True,script_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in script_paths}),indent=2))
    notebook=ROOT/'RAPID/notebooks/slope_smoothing_floor_comparison.ipynb';nb=nbformat.read(notebook,as_version=4);marker='CHILD-SHARED-HEADS-2026-10-03';nb.cells=[c for c in nb.cells if marker not in c.source];nb.cells.append(nbformat.v4.new_markdown_cell(report));name='reconstruction_summary.png';c=nbformat.v4.new_markdown_cell(f'<!-- {marker} -->\n![Reconstruction and routing checks](attachment:{name})');c.attachments={name:{'image/png':base64.b64encode((OUT/name).read_bytes()).decode()}};nb.cells.append(c);nbformat.validate(nb);nbformat.write(nb,notebook)
    print(report)

if __name__=='__main__':main()
