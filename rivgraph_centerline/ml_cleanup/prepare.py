"""Audit and sample original/prepared/cleaned pairs without rescaling geometry."""
import hashlib
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from core import ROOT, OUT, CORE, HALO, SIDE, CROP, SEED, aligned_offset, dump


def main():
    (OUT/'tiles').mkdir(parents=True,exist_ok=True)
    audit=[]; records=[]
    for d in sorted(ROOT.iterdir()):
        c=d/'masks_cleaned'/f'{d.name}_cleaned.tif'; b=d/'masks_prepared'/f'{d.name}_binary_projected.tif'
        if not(c.exists() and b.exists()): continue
        row={'river':d.name}; audit.append(row)
        prep=d/'prepare_summary.json'; edits=d/'manual_edits_summary.json'
        if not(prep.exists() and edits.exists()):
            row['excluded']='Missing preparation or manual-edit provenance';continue
        meta=json.loads(prep.read_text()); src=Path(meta['source_mask'])
        if not src.exists(): row['excluded']='Original source unavailable';continue
        summary=json.loads(edits.read_text())
        row.update(source=str(src), threshold=meta['threshold'], actions=summary.get('actions'),
                   automated_component_filter=summary.get('largest_water_component_filter'),
                   automated_hole_cleanup=summary.get('main_river_small_land_cleanup'))
        rng=np.random.default_rng(SEED+int(d.name))
        with rasterio.open(b) as bs, rasterio.open(c) as cs, rasterio.open(src) as ss:
            coloff,rowoff=aligned_offset(bs,cs)
            row.update(base_shape=list(bs.shape),clean_shape=list(cs.shape),offset=[coloff,rowoff],pixel_size=abs(bs.res[0]),source_dtype=ss.dtypes[0],source_nodata=ss.nodata)
            # Reprojection matches the base grid; continuous values use bilinear interpolation.
            with WarpedVRT(ss, crs=bs.crs, transform=bs.transform,width=bs.width,height=bs.height,resampling=Resampling.bilinear,src_nodata=None,nodata=0) as raw:
                # Evaluate only within cleaned extent, with input-only selection.
                grid=[(r,k) for r in range(HALO,cs.height-CORE-HALO+1,CORE) for k in range(HALO,cs.width-CORE-HALO+1,CORE)]
                rng.shuffle(grid)
                selected=[]; seen=set(); tried=0
                for r,k in grid:
                    tried+=1
                    patch=bs.read(1,window=Window(k+coloff,r+rowoff,CORE,CORE))
                    if np.count_nonzero(patch)/patch.size >= .01:
                        selected.append((r,k,'natural'));seen.add((r,k))
                    if len(selected)>=40: break
                row['natural_candidates_examined']=tried
                # Challenge patches are labelled separately and never pooled as representative area.
                gp=d/'manual_edits'/f'{d.name}_manual_edits.gpkg'
                if gp.exists():
                    g=gpd.read_file(gp).to_crs(cs.crs)
                    for action in ['add','remove']:
                        subset=g[g.action.astype(str).str.lower()==action]
                        indices=rng.permutation(len(subset)); count=0
                        for idx in indices:
                            geom=subset.iloc[idx].geometry
                            if geom is None or geom.is_empty:continue
                            point=geom.representative_point(); rr,kk=cs.index(point.x,point.y)
                            r=int(np.clip(rr-CORE//2,HALO,cs.height-CORE-HALO));k=int(np.clip(kk-CORE//2,HALO,cs.width-CORE-HALO))
                            if (r,k) in seen:continue
                            selected.append((r,k,'edit_'+action));seen.add((r,k));count+=1
                            if count>=8:break
                for i,(r,k,stratum) in enumerate(selected):
                    w=Window(k-HALO,r-HALO,SIDE,SIDE);bw=Window(k-HALO+coloff,r-HALO+rowoff,SIDE,SIDE)
                    base=bs.read(1,window=bw); target=cs.read(1,window=w); original=raw.read(1,window=bw,out_dtype='float32')
                    if not(set(np.unique(base))<={0,1} and set(np.unique(target))<={0,1}):raise ValueError('Non-binary label')
                    # Fixed threshold-relative scaling supports uint8 water scores and binary inputs.
                    original=np.clip(original/max(float(meta['threshold']),1e-6),0,5)/5
                    f=f'tiles/{d.name}_{i:03d}.npz'
                    np.savez_compressed(OUT/f,base=base.astype(bool),target=target.astype(bool),raw=original.astype(np.float32))
                    records.append(dict(river=d.name,file=f,stratum=stratum,row=r,col=k,pixel_size=abs(bs.res[0]),
                                        transform=list(cs.window_transform(w)),crs=str(cs.crs),
                                        changed=int(np.count_nonzero(base[CROP]!=target[CROP])),
                                        sha256=hashlib.sha256((OUT/f).read_bytes()).hexdigest()))
        row['tiles']=sum(x['river']==d.name for x in records)
        dump(OUT/'audit.json',audit);dump(OUT/'tiles.json',records)
        print(d.name,row['tiles'],'tiles',flush=True)
    dump(OUT/'audit.json',audit)
    ids=sorted({r['river'] for r in records});np.random.default_rng(SEED).shuffle(ids)
    groups=[ids[i::4] for i in range(4)]
    folds=[]
    for i in range(4):
        test=groups[i];val=groups[(i+1)%4];train=[x for x in ids if x not in test+val]
        folds.append(dict(fold=i,train=train,validation=val,test=test))
    dump(OUT/'splits.json',folds)
    print('Prepared',len(records),'tiles;',len(ids),'rivers',flush=True)

if __name__=='__main__':main()
