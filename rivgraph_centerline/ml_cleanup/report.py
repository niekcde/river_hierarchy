"""Aggregate held-out predictions with river-level uncertainty and review images."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import OUT,CROP,SEED,dump,load_tile,scores,counts

METHODS=['unchanged','rules','trees_binary','trees_raw','unet','unet_conservative']
LABELS=dict(unchanged='Unchanged mask',rules='Small-object/hole rules',trees_binary='Image trees: binary',trees_raw='Image trees: original + binary',unet='U-Net',unet_conservative='U-Net: conservative edits')
META={'fold','method','tile','river','stratum','threshold'}

def aggregate(rows):
    c={k:sum(r[k] for r in rows) for k in rows[0] if k not in META}
    return scores(c)

def number(v):return '—' if v is None else f'{v:.3f}'

def main():
    metrics=[]
    for p in sorted(OUT.glob('fold_*/metrics.json')):metrics+=json.loads(p.read_text())
    records={r['file']:r for r in json.loads((OUT/'tiles.json').read_text())}
    per=[];summary=[]
    for method in METHODS:
        for stratum in ['natural','edit_add','edit_remove']:
            rr=[r for r in metrics if r['method']==method and r['stratum']==stratum]
            if not rr:continue
            rs=[]
            for river in sorted({r['river'] for r in rr}):
                s=aggregate([r for r in rr if r['river']==river]);rs.append(s);per.append(dict(method=method,stratum=stratum,river=river,**s))
            macro={k:float(np.mean([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else None for k in rs[0]}
            summary.append(dict(method=method,stratum=stratum,rivers=len(rs),macro=macro,pooled=aggregate(rr)))
    dump(OUT/'per_river.json',per);dump(OUT/'summary.json',summary)
    rng=np.random.default_rng(SEED);comparisons=[]
    for method in METHODS[1:]:
        a={r['river']:r for r in per if r['method']==method and r['stratum']=='natural'}
        b={r['river']:r for r in per if r['method']=='unchanged' and r['stratum']=='natural'}
        if not a:continue
        for metric in ['iou','boundary_f1','cldice','damage_rate','island_iou']:
            diff=np.array([a[n][metric]-b[n][metric] for n in a if a[n][metric] is not None and b[n][metric] is not None])
            if not len(diff):continue
            boot=np.mean(rng.choice(diff,(5000,len(diff)),replace=True),axis=1)
            comparisons.append(dict(method=method,metric=metric,mean_difference=float(diff.mean()),ci95=list(np.quantile(boot,[.025,.975])),rivers=len(diff),improved=int((diff<0).sum()) if metric=='damage_rate' else int((diff>0).sum())))
    dump(OUT/'paired_bootstrap.json',comparisons)
    # Threshold sensitivity uses saved scores only, with no model reselection.
    sensitivity=[]
    for threshold in [.2,.35,.5,.65,.8]:
        sums=defaultdict(lambda:[0,0,0,0])
        for r in [r for r in metrics if r['method']=='unet' and r['stratum']=='natural']:
            rec=records[r['tile']];z=load_tile(rec)
            path=OUT/f"fold_{r['fold']}"/'predictions'/'unet'/(Path(r['tile']).stem+'.npz')
            with np.load(path) as f:p=f['probability']>=threshold
            y=z['target'][CROP];b=z['base'][CROP];a=sums[r['river']]
            a[0]+=int((p&y).sum());a[1]+=int((p|y).sum());a[2]+=int(((p!=b)&(b==y)).sum());a[3]+=int((b==y).sum())
        if sums:sensitivity.append(dict(threshold=threshold,macro_iou=float(np.mean([v[0]/v[1] for v in sums.values()])),macro_damage=float(np.mean([v[2]/v[3] for v in sums.values()]))))
    dump(OUT/'threshold_sensitivity.json',sensitivity)
    # Best and worst changes are deliberately both shown.
    u=[r for r in metrics if r['method']=='unet'];base={r['tile']:r for r in metrics if r['method']=='unchanged'}
    if u:
        ranked=sorted([r for r in u if r['stratum']=='natural'],key=lambda r:r['errors']-base[r['tile']]['errors'])
        chosen=ranked[:2]+ranked[-2:]
        for stratum in ['edit_add','edit_remove']:
            choices=[r for r in u if r['stratum']==stratum]
            if choices:chosen.append(max(choices,key=lambda r:r['errors']))
        fig,axs=plt.subplots(len(chosen),5,figsize=(14,2.8*len(chosen)))
        for i,r in enumerate(chosen):
            z=load_tile(records[r['tile']]);pfile=OUT/f"fold_{r['fold']}"/'predictions'/'unet'/(Path(r['tile']).stem+'.npz')
            with np.load(pfile) as f:p=f['prediction']
            y=z['target'][CROP];err=np.zeros((*y.shape,3));err[p&~y]=[.9,.15,.15];err[~p&y]=[.1,.45,1];err[p&y]=[.7,.7,.7]
            for ax,arr,title in zip(axs[i],[z['raw'][CROP],z['base'][CROP],y,p,err],['Original values','Prepared mask','Manual target','Held-out U-Net','Error: red extra / blue missing']):
                ax.imshow(arr,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
            axs[i,0].set_ylabel(f"River {r['river']}\n{r['stratum']}");axs[i,0].set_title(f"River {r['river']} · {r['stratum']}\nOriginal values",fontsize=9)
        fig.tight_layout();fig.savefig(OUT/'held_out_examples.png',dpi=160);plt.close(fig)
    natural=[r for r in summary if r['stratum']=='natural']
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,key,title in zip(axs,['iou','boundary_f1','cldice'],['Water overlap (IoU)','Bank boundary F1 (2 px)','Skeleton connectivity (clDice)']):
        for i,m in enumerate(METHODS):
            yy=[r[key] for r in per if r['method']==m and r['stratum']=='natural' and r[key] is not None]
            if yy:ax.scatter(np.full(len(yy),i),yy,s=16,alpha=.6);ax.plot([i-.2,i+.2],[np.mean(yy)]*2,color='black',lw=3)
        ax.set_xticks(range(6),['Original','Rules','Trees\nbinary','Trees\nraw','U-Net','Conservative'],rotation=35);ax.set_ylim(0,1.03);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'river_scores.png',dpi=150);plt.close(fig)
    lines=['# River-mask cleaning: held-out pilot','',
      'This is an experimental feasibility benchmark, not a production replacement for reviewed masks. Original/prepared/cleaned inputs were not modified.','',
      '## Data and design','',
      '- 17 paired prepared/cleaned masks found; 15 used. Run 41 lacks a manual-edit summary; run 56 has no accessible original source at its recorded path.',
      '- 840 native-resolution tiles: 40 input-water-selected tiles plus 8 add-edit and 8 remove-edit locations per river. Each has a 192 × 192 scoring core and 32-pixel context on every side.',
      '- All 840 cached pairs were checked exactly against their georeferenced source rasters. Eligible river bounding boxes do not overlap.',
      '- Four fixed outer folds; every river is tested once. Each fold uses disjoint training, validation and test rivers. Validation chooses U-Net checkpoint and thresholds; test results do not guide training.',
      '- Prepared/cleaned raster zeros are valid land despite their nodata metadata. Cropped target boundaries are aligned using their affine transforms. Training and tile evaluation stay inside the available cleaned extent.',
      '- Original source values are reprojected bilinearly and scaled by the recorded source threshold, clipped to five times that threshold. These values are not assumed to be calibrated water probabilities.',
      '- Image-feature models are Extra Trees with local densities, distance transforms and component sizes. Binary-only versus original-valued features is an input ablation. U-Net uses binary and original-valued channels, boundary/edit-weighted cross-entropy, Dice and soft skeleton loss.',
      '- U-Net is a small residual model with rotations/flips, up to 18 epochs of 192 sampled tiles, early stopping after six unimproved validation epochs. This is a bounded pilot, not an exhaustive architecture or training-budget search.',
      '- The simple baseline removes water components below 16 pixels and fills enclosed holes up to 3 pixels using context tiles. It is local cleanup, not a global main-river selector.','',
      '## Primary results','',
      'Each river has equal weight. These are sampled source-water-containing tiles inside the cleaned extent, not whole-river or whole-floodplain accuracy.','',
      '| Method | IoU ↑ | Boundary F1 ↑ | clDice ↑ | Removal recall ↑ | Addition recall ↑ | Edit precision ↑ | Damage rate ↓ |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in natural:
        a=r['macro'];lines.append('| '+LABELS[r['method']]+' | '+' | '.join(number(a[k]) for k in ['iou','boundary_f1','cldice','removal_recall','addition_recall','edit_precision'])+f" | {100*a['damage_rate']:.3f}% |")
    lines += ['','Damage rate is the fraction of initially correct pixels changed incorrectly. Edit precision is the fraction of proposed edits agreeing with the cleaned target; undefined values are omitted from macro averages. Boundary matches allow two pixels, whose ground distance varies by river. clDice measures skeleton overlap and is not a guarantee of correct river-network topology.','',
      '![Per-river scores](results/river_scores.png)','',
      '## Paired uncertainty','',
      '95% bootstrap intervals resample whole rivers (5,000 replicates). They describe variation across these rivers, not training-seed uncertainty.','',
      '| Method | Mean IoU change | 95% interval | Rivers improved |','|---|---:|---:|---:|']
    for c in comparisons:
        if c['metric']=='iou':lines.append(f"| {LABELS[c['method']]} | {c['mean_difference']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {c['improved']}/{c['rivers']} |")
    lines+=['','## Saved-edit challenge locations','','These are deliberately difficult, label-selected locations and are reported separately. Add/remove actions do not tell us whether an edit was a wetland, island, bank, or scope decision.','',
      '| Location | Method | IoU | Addition recall | Removal recall | Edit precision |','|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['stratum']=='natural':continue
        a=r['macro'];lines.append('| '+r['stratum']+' | '+LABELS[r['method']]+' | '+' | '.join(number(a[k]) for k in ['iou','addition_recall','removal_recall','edit_precision'])+' |')
    lines+=['','## Islands and connectivity limitations','',
      'Island metrics count enclosed 4-connected land regions of at least 4 pixels; water uses 8-connectivity. Islands intersecting a tile boundary are excluded. Per-tile component counts and island-area IoU are saved in the machine-readable results. These cannot establish whole-network connectivity or identity-preserving island matching.','',
      'The target combines manual decisions and automatic component/hole cleanup. A local model may not infer which distant water body belongs to the intended main river. Unusual island shape alone is insufficient evidence of an error. No explicit wetland/island semantic labels or independent expert rereview were available.','',
      '## Visual review','',
      'The examples include the two largest improvements, two largest regressions, and hardest add/remove edit examples, measured by pixel error. They are selected for diagnosis, not as representative samples.','',
      '![Held-out examples](results/held_out_examples.png)','',
      '## Reproducibility and artifacts','',
      'See [README.md](README.md) for commands. `results/audit.json`, `integrity.json`, `tiles.json`, `splits.json`, and `environment.json` record data, checks and experiment setup. Each fold saves models, training history, validation selection, exact binary predictions and pixel counts. `summary.json`, `per_river.json`, `paired_bootstrap.json`, and `threshold_sensitivity.json` contain numerical results.','',
      'Prediction scores are uncalibrated; conservative cutoffs do not establish a confidence guarantee. Threshold-sensitivity calculations use stored float16 scores. Original sources remain external dependencies.','']
    (OUT.parent/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(natural,indent=2))

if __name__=='__main__':main()
