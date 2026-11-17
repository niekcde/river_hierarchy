"""Validate cached pairs against their source files and check geographic overlap."""
import hashlib,json
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from core import ROOT,OUT,HALO,SIDE,CROP,aligned_offset,dump,load_tile

def main():
    audit=json.loads((OUT/'audit.json').read_text());records=json.loads((OUT/'tiles.json').read_text())
    ids={x['river'] for x in audit}
    for d in sorted(ROOT.iterdir()):
        c=d/'masks_cleaned'/f'{d.name}_cleaned.tif';b=d/'masks_prepared'/f'{d.name}_binary_projected.tif'
        if c.exists() and b.exists() and d.name not in ids:
            audit.append(dict(river=d.name,excluded='Original source unavailable' if (d/'manual_edits_summary.json').exists() else 'Missing manual-edit provenance'))
    checked=0
    for a in audit:
        if 'excluded' in a:continue
        n=a['river'];d=ROOT/n
        with rasterio.open(d/'masks_prepared'/f'{n}_binary_projected.tif') as b,rasterio.open(d/'masks_cleaned'/f'{n}_cleaned.tif') as c:
            co,ro=aligned_offset(b,c);a['geographic_bounds']=list(transform_bounds(c.crs,'EPSG:4326',*c.bounds,densify_pts=21))
            for r in [r for r in records if r['river']==n]:
                z=load_tile(r);w=Window(r['col']-HALO,r['row']-HALO,SIDE,SIDE)
                assert np.array_equal(z['target'],c.read(1,window=w)>0)
                w=Window(r['col']-HALO+co,r['row']-HALO+ro,SIDE,SIDE)
                assert np.array_equal(z['base'],b.read(1,window=w)>0)
                assert hashlib.sha256((OUT/r['file']).read_bytes()).hexdigest()==r['sha256']
                checked+=1
    overlaps=[]
    valid=[a for a in audit if 'excluded' not in a]
    for i,a in enumerate(valid):
        for b in valid[i+1:]:
            x=a['geographic_bounds'];y=b['geographic_bounds']
            if max(x[0],y[0])<min(x[2],y[2]) and max(x[1],y[1])<min(x[3],y[3]):overlaps.append([a['river'],b['river']])
    dump(OUT/'audit.json',audit);dump(OUT/'integrity.json',dict(cached_pairs_verified=checked,geographic_bbox_overlaps=overlaps))
    print('Verified',checked,'cached pairs; geographic overlaps:',overlaps)
if __name__=='__main__':main()
