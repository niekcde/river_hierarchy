"""Additional label-based evaluation strata; these never become model inputs."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
from core import OUT,CROP,load_tile,dump,holes


def main():
    rec={r['file']:r for r in json.loads((OUT/'tiles.json').read_text())}
    rows=[]
    for folder in sorted(OUT.glob('fold_*')):
        if not (folder/'DONE.json').exists():continue
        for r in json.loads((folder/'metrics.json').read_text()):
            if r['stratum']!='natural':continue
            z=load_tile(rec[r['tile']]);y=z['target'][CROP];b=z['base'][CROP]
            with np.load(folder/'predictions'/r['method']/(Path(r['tile']).stem+'.npz')) as f:p=f['prediction']
            # Near-channel area includes land within 16 pixels of retained water.
            distance=ndi.distance_transform_edt(~z['target'])
            near=(distance<=16)[CROP] if z['target'].any() else np.zeros(y.shape,bool)
            hy,nh=holes(y);hp,_=holes(p)
            lab,n=ndi.label(hy);preserved=0
            plab,_=ndi.label(hp)
            for k in range(1,n+1):
                obj=lab==k;matches=np.unique(plab[obj]);matches=matches[matches!=0]
                if any(np.count_nonzero(obj&(plab==j))/np.count_nonzero(obj|(plab==j))>=.5 for j in matches):preserved+=1
            rows.append(dict(river=r['river'],method=r['method'],tile=r['tile'],
                 near_tp=int((p&y&near).sum()),near_union=int(((p|y)&near).sum()),
                 near_damage=int(((p!=b)&(b==y)&near).sum()),near_correct=int(((b==y)&near).sum()),
                 true_islands=nh,preserved_islands=preserved,
                 retained_water=int((b&y).sum()),lost_retained_water=int((b&y&~p).sum()),
                 offchannel_water_before=int((b&~near).sum()),offchannel_water_after=int((p&~near).sum())))
    summaries=[]
    for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['method']==method];ss=[]
        for river in sorted({r['river'] for r in rr}):
            a=[r for r in rr if r['river']==river];s={k:sum(r[k] for r in a) for k in a[0] if k not in ['method','river','tile']}
            ss.append(dict(river=river,near_channel_iou=s['near_tp']/s['near_union'] if s['near_union'] else None,
                near_channel_damage=s['near_damage']/s['near_correct'] if s['near_correct'] else None,
                island_object_recall=s['preserved_islands']/s['true_islands'] if s['true_islands'] else None,
                retained_water_loss=s['lost_retained_water']/s['retained_water'] if s['retained_water'] else None,**s))
        macro={k:float(np.mean([s[k] for s in ss if s[k] is not None])) for k in ['near_channel_iou','near_channel_damage','island_object_recall','retained_water_loss'] if any(s[k] is not None for s in ss)}
        summaries.append(dict(method=method,macro=macro,per_river=ss))
    dump(OUT/'bank_island_diagnostics.json',summaries)
    print(json.dumps([{k:v for k,v in r.items() if k!='per_river'} for r in summaries],indent=2))
if __name__=='__main__':main()
