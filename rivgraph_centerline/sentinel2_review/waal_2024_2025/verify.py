"""Independent checks of acquisition completeness and geospatial review products."""
import json,hashlib
from pathlib import Path
import numpy as np,rasterio,geopandas as gpd
from fetch import ROOT,TRANSFORM,WIDTH,HEIGHT
OUT=ROOT/'review'

def main():
    inv=json.loads((ROOT/'inventory.json').read_text());catalog=json.loads((ROOT/'catalog.json').read_text())
    assert not inv['errors'] and len(inv['results'])==len(catalog)
    assert len({r['date'] for r in inv['results']})==len(catalog)
    count=np.zeros((HEIGHT,WIDTH),np.uint16);numerator=np.zeros_like(count)
    for r in inv['results']:
        assert '2024-01-01'<=r['date']<='2025-12-31'
        if r['status']=='downloaded':
            with np.load(ROOT/'cache'/f"{r['date']}.npz") as z:
                assert z['valid'].sum()==r['valid_pixels'];count+=z['valid'];numerator+=(z['water']&z['valid'])
    def read(name):
        with rasterio.open(OUT/(name+'.tif')) as r:return r.read(1)
    np.testing.assert_array_equal(count,read('valid_observation_count'))
    support=count>=5;occ=read('spectral_water_occurrence');np.testing.assert_allclose(occ[support],numerator[support]/count[support],atol=1e-7)
    assert (occ[~support]==-9999).all()
    allmask=read('water_mask_occ75_all');main=read('water_mask_occ75_largest_component');excluded=read('excluded_components_retained')
    assert np.array_equal(allmask[support],(occ[support]>=.75).astype('uint8'))
    assert np.array_equal(allmask[support],main[support]+excluded[support])
    assert not((main==1)&(excluded==1)).any()
    hashes={}
    for f in OUT.glob('*.tif'):
        with rasterio.open(f) as r:
            assert r.shape==(HEIGHT,WIDTH) and r.transform==TRANSFORM and r.crs.to_epsg()==32631
            assert np.isfinite(r.read()).all()
        hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    g=gpd.read_file(OUT/'review_annotations.gpkg',layer='review_tiles')
    assert len(g)==1 and(g.status=='pending').all()
    assert len(gpd.read_file(OUT/'review_annotations.gpkg',layer='manual_edits'))==0
    result=dict(dates_verified=len(catalog),raster_files_verified=len(hashes),exact_observation_count_check=True,occurrence_check=True,component_reconstruction_check=True,annotation_status='pending',sha256=hashes)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
