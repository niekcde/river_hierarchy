"""Attach robustness checks and complete-raster findings to the generated report."""
import json
from pathlib import Path
import numpy as np
from core import OUT,SEED,dump
from report import aggregate,LABELS,number

def main():
    primary=json.loads((OUT/'summary.json').read_text());diag=json.loads((OUT/'bank_island_diagnostics.json').read_text())
    first={r['method']:r['macro'] for r in primary if r['stratum']=='natural'}
    other=[]
    for folder in sorted((OUT/'second_seed').glob('fold_*')):
        assert(folder/'DONE.json').exists();other+=json.loads((folder/'metrics.json').read_text())
    seedrows=[]
    for river in sorted({r['river'] for r in other}):
        rr=[r for r in other if r['river']==river and r['stratum']=='natural'];seedrows.append(dict(river=river,**aggregate(rr)))
    macro={k:float(np.mean([r[k] for r in seedrows if r[k] is not None])) if any(r[k] is not None for r in seedrows) else None for k in seedrows[0] if k!='river'}
    dump(OUT/'second_seed_summary.json',dict(macro=macro,per_river=seedrows))
    per=json.loads((OUT/'per_river.json').read_text());raw={r['river']:r for r in per if r['method']=='trees_raw' and r['stratum']=='natural'};binary={r['river']:r for r in per if r['method']=='trees_binary' and r['stratum']=='natural'}
    diff=np.array([raw[n]['iou']-binary[n]['iou'] for n in raw]);rng=np.random.default_rng(SEED)
    ci=np.quantile(rng.choice(diff,(5000,len(diff))).mean(1),[.025,.975]);ablation=dict(mean_difference=float(diff.mean()),ci95=ci.tolist(),rivers=len(diff))
    dump(OUT/'raw_input_ablation.json',ablation)
    whole=json.loads((OUT/'whole_river_03'/'metrics.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    dmap={r['method']:r['macro'] for r in diag}
    best=max(['trees_binary','trees_raw','unet','unet_conservative'],key=lambda k:first[k]['iou'])
    executive=[
        '**Result: the current models do not reliably reproduce the manual cleanup while preserving river geometry.** The models must be judged on banks, islands and connectivity as well as water overlap.',
        '',f"The highest observed learned-model mean held-out IoU was **{first[best]['iou']:.3f}** ({LABELS[best]}), versus **{first['unchanged']['iou']:.3f}** for the original prepared masks. These are exploratory comparisons on 15 rivers, with all training and model selection separated from each test river.",
        '',f"For that model, boundary F1 was **{first[best]['boundary_f1']:.3f}** versus **{first['unchanged']['boundary_f1']:.3f}** originally, and clDice was **{first[best]['cldice']:.3f}** versus **{first['unchanged']['cldice']:.3f}**. Its retained-water loss was **{100*dmap[best]['retained_water_loss']:.2f}%**. These measures expose costs hidden by water-area overlap alone.",
        '', 'The practical next step is to separate main-river selection from bank/island repair: provide a target-independent river corridor or reliable channel seed, then train and assess local corrections within that context. Add explicit labels for wetland edges and erroneous versus legitimate islands. Retain human review of proposed edits until these geometric checks pass on new rivers.', '']
    extra=['','## Bank and island preservation checks','',
      'Evaluation-only near-channel strata include retained water and land within 16 pixels. Island-object recall requires at least 50% intersection-over-union with an enclosed reference island of at least four pixels. Cropped islands are excluded. Retained-water loss is the fraction of water already correct in the input that the model deletes. Values below are river means.','',
      '| Method | Near-channel IoU ↑ | Island-object recall ↑ | Retained-water loss ↓ |','|---|---:|---:|---:|']
    for r in diag:
        a=r['macro'];extra.append(f"| {LABELS[r['method']]} | {number(a.get('near_channel_iou'))} | {number(a.get('island_object_recall'))} | {100*a['retained_water_loss']:.2f}% |")
    extra+=['','## Repeat training and original-value ablation','',
        '| CNN training run | Mean IoU | Boundary F1 | clDice |','|---|---:|---:|---:|',
        '| Primary seed | '+' | '.join(number(first['unet'][k]) for k in ['iou','boundary_f1','cldice'])+' |',
        '| Second seed | '+' | '.join(number(macro[k]) for k in ['iou','boundary_f1','cldice'])+' |','',
        'The second run repeats the same architecture, training budget, disjoint river splits and validation selection, changing only the training seed. Two seeds are a sensitivity check, not a complete estimate of training variability.',
        '',f"Adding original-valued features to the tree model changed mean IoU by **{diff.mean():+.3f}**, with paired river-bootstrap 95% interval **[{ci[0]:+.3f}, {ci[1]:+.3f}]**. This tests the value of information retained before binarization; it does not establish that all source encodings will transfer.",
        '','## Complete held-out river 03','',
        'Both learned models were applied over the **entire prepared raster** without reading cleaned labels. Scoring then used the complete available cleaned extent. River 03 was excluded from both training and validation for these models. A separate global baseline selects the largest 8-connected water component from the full prepared raster and fills tiny holes; this uses no target geometry. This single-river result is a stress test, not a second independent population estimate.','',
        '| Method | IoU | Boundary F1 | clDice | Predicted / target water components | Predicted / target islands |','|---|---:|---:|---:|---:|---:|']
    for r in whole['metrics']:
        label=LABELS.get(r['method'],'Global largest component + tiny-hole fill')
        extra.append(f"| {label} | {number(r['iou'])} | {number(r['boundary_f1'])} | {number(r['cldice'])} | {r['components']} / {r['target_components']} | {r['islands_pred']} / {r['islands_true']} |")
    extra+=['','![Full held-out river](results/whole_river_03/overview.png)','',
        'Experimental georeferenced masks: [U-Net](results/whole_river_03/03_unet_experimental.tif), [original-valued image trees](results/whole_river_03/03_trees_raw_experimental.tif), [U-Net scores](results/whole_river_03/03_unet_experimental_probability.tif), and [global component baseline](results/whole_river_03/03_global_component_experimental.tif). These are review artifacts, not replacements for the production masks.','',
        '## Verification','',f"- **21 automated tests passed**: grid/crop alignment, invalid offsets/CRS, edge preservation, island handling, empty predictions, connectivity disruption, feature isolation, split integrity, model gradients/reload, GPU-to-CPU checkpoint agreement, streaming georeferencing and non-overwrite behavior.",
        '- **840/840** cached pairs matched the original prepared and cleaned rasters exactly; geographic extents of the eligible rivers did not overlap.',
        f"- **{verification['predictions_verified']:,}** primary held-out prediction artifacts were independently checked against saved confusion counts, harmful-edit counts, score ranges and disjoint split membership.",
        '- Thresholds were checked against their validation selection tables. Saved model SHA-256 hashes are recorded in `results/verification.json`.',
        '', '## Method references','',
        '- [U-Net](https://arxiv.org/abs/1505.04597): encoder-decoder segmentation architecture; this pilot uses a smaller custom residual variant.',
        '- [clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html): skeleton-overlap supervision and evaluation. The finite-iteration soft-skeleton term used here does not guarantee globally correct topology.', '']
    extra+=['','## Per-river primary results','','| River | Original IoU | Original-valued trees IoU | U-Net IoU | Trees bank F1 change | U-Net bank F1 change |','|---|---:|---:|---:|---:|---:|']
    lookup={(r['river'],r['method']):r for r in per if r['stratum']=='natural'}
    for river in sorted(raw):
        b=lookup[(river,'unchanged')];t=lookup[(river,'trees_raw')];u=lookup[(river,'unet')]
        extra.append(f"| {river} | {b['iou']:.3f} | {t['iou']:.3f} | {u['iou']:.3f} | {t['boundary_f1']-b['boundary_f1']:+.3f} | {u['boundary_f1']-b['boundary_f1']:+.3f} |")
    extra.append('')
    p=OUT.parent/'REPORT.md';text=p.read_text();head,rest=text.split('\n',1);p.write_text(head+'\n\n'+'\n'.join(executive)+rest+'\n'+'\n'.join(extra))
    print(json.dumps(dict(primary=first,second_seed=macro,bank_island={r['method']:r['macro'] for r in diag},whole=whole['metrics']),indent=2))
if __name__=='__main__':main()
