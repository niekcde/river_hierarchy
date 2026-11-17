"""Download aligned Sentinel-2 Collection-1 subsets for an independently reviewed tile."""
import os
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS','.tif')
os.environ.setdefault('GDAL_HTTP_MAX_RETRY','3')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY','2')
import argparse,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests,numpy as np,rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage as ndi

ROOT=Path(__file__).resolve().parent
BBOX=[5.57,51.86,5.66,51.91]
BANDS=['red','green','blue','nir','swir16']
COLLECTION='sentinel-2-c1-l2a'

def grid():
    x0,y0,x1,y1=transform_bounds('EPSG:4326','EPSG:32631',*BBOX)
    x0=np.floor(x0/10)*10;y0=np.floor(y0/10)*10;x1=np.ceil(x1/10)*10;y1=np.ceil(y1/10)*10
    return from_origin(x0,y1,10,10),int((x1-x0)/10),int((y1-y0)/10)

TRANSFORM,WIDTH,HEIGHT=grid()

def catalog():
    path=ROOT/'catalog.json'
    if path.exists():return json.loads(path.read_text())
    j=json.loads((ROOT/'catalog_page1.json').read_text());features=j['features']
    while True:
        link=next((x for x in j['links'] if x['rel']=='next'),None)
        if not link:break
        r=requests.get(link['href'],timeout=60);r.raise_for_status();j=r.json();features+=j['features']
    # A single MGRS tile covers the complete AOI. Keep one item per UTC date.
    chosen={}
    for f in features:
        if f['properties']['grid:code']!='MGRS-31UFT':continue
        day=f['properties']['datetime'][:10]
        if day not in chosen or f['properties']['eo:cloud_cover']<chosen[day]['properties']['eo:cloud_cover']:chosen[day]=f
    result=sorted(chosen.values(),key=lambda f:f['properties']['datetime'])
    path.write_text(json.dumps(result));return result


def read_asset(asset,kind):
    with rasterio.open(asset['href']) as src:
        with WarpedVRT(src,crs='EPSG:32631',transform=TRANSFORM,width=WIDTH,height=HEIGHT,
                resampling=Resampling.nearest if kind=='scl' else Resampling.bilinear,
                dtype='float32' if kind!='scl' else 'uint8',nodata=0) as vrt:
            dn=vrt.read(1)
    if kind=='scl':return dn
    b=asset['raster:bands'][0]
    # Collection-1 explicitly retains the BOA encoding offset; apply metadata once.
    return np.where(dn>0,dn*float(b['scale'])+float(b.get('offset',0)),np.nan).astype('float32')


def fetch(f):
    day=f['properties']['datetime'][:10];out=ROOT/'cache'/f'{day}.npz';meta=out.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    scl=read_asset(f['assets']['scl'],'scl')
    cloud=ndi.binary_dilation(np.isin(scl,[2,3,8,9,10,11]),iterations=4)
    valid=np.isin(scl,[4,5,6])&~cloud
    row=dict(date=day,item=f['id'],scene_cloud_percent=f['properties']['eo:cloud_cover'],scl_valid_fraction=float(valid.mean()))
    if not valid.any():
        row.update(status='no_usable_pixels',valid_pixels=0);meta.write_text(json.dumps(row));return row
    data={b:read_asset(f['assets'][b],b) for b in BANDS}
    valid &= np.logical_and.reduce([np.isfinite(a) for a in data.values()])
    data={k:np.maximum(a,0) for k,a in data.items()}
    denom=data['green']+data['swir16'];valid &= denom>1e-6
    mndwi=np.clip((data['green']-data['swir16'])/np.maximum(denom,1e-6),-1,1)
    ndvi=np.clip((data['nir']-data['red'])/np.maximum(data['nir']+data['red'],1e-6),-1,1)
    water=(mndwi>0)&(ndvi<.3)&valid
    row.update(status='downloaded',valid_pixels=int(valid.sum()),valid_fraction=float(valid.mean()),spectral_water_pixels=int(water.sum()))
    np.savez_compressed(out,**data,scl=scl,valid=valid,water=water,mndwi=mndwi,ndvi=ndvi)
    meta.write_text(json.dumps(row));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');args=p.parse_args()
    (ROOT/'cache').mkdir(exist_ok=True);items=catalog()
    (ROOT/'config.json').write_text(json.dumps(dict(site='Waal near Dodewaard/Druten, Netherlands',bbox=BBOX,crs='EPSG:32631',transform=list(TRANSFORM),width=WIDTH,height=HEIGHT,start='2024-01-01',end='2025-12-31',collection=COLLECTION,minimum_observations=5,water_rule='MNDWI > 0 AND NDVI < 0.3',composite_threshold=.75,valid_scl=[4,5,6],cloud_shadow_snow_buffer_m=40,original_classifier_available=False),indent=2))
    if args.preview:
        items=sorted([f for f in items if f['properties']['datetime'][5:7] in ['06','07','08']],key=lambda f:f['properties']['eo:cloud_cover'])[:1]
    results=[];errors=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(fetch,f):f['id'] for f in items}
        for i,future in enumerate(as_completed(futures),1):
            try:
                r=future.result();results.append(r);print(i,'/',len(items),r['date'],r['status'],r['valid_pixels'],flush=True)
            except Exception as e:
                errors.append(dict(item=futures[future],error=str(e)));print('FAILED',futures[future],str(e),flush=True)
            (ROOT/('preview_inventory.json' if args.preview else 'inventory.json')).write_text(json.dumps(dict(results=sorted(results,key=lambda r:r['date']),errors=errors),indent=2))
    if errors:raise RuntimeError(f'{len(errors)} scene downloads failed; rerun to retry without redownloading cached scenes')

if __name__=='__main__':main()
