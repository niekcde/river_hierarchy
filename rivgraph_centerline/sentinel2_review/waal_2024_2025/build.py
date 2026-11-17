"""Build a two-year Sentinel-2 review package; scores are not calibrated probabilities."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/river-mask-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/river-mask-cache')
import json,warnings,hashlib
from pathlib import Path
import numpy as np,pandas as pd,geopandas as gpd,rasterio
from rasterio.windows import Window
from shapely.geometry import box
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fetch import ROOT,TRANSFORM,WIDTH,HEIGHT

OUT=ROOT/'review';OUT.mkdir(exist_ok=True)
NODATA=-9999.
FOCUS=Window(210,80,250,140)

def write(name,a,nodata=NODATA,descriptions=None,valid=None):
    if a.ndim==2:a=a[None]
    profile=dict(driver='GTiff',width=WIDTH,height=HEIGHT,count=a.shape[0],dtype=str(a.dtype),crs='EPSG:32631',transform=TRANSFORM,nodata=nodata,compress='deflate',tiled=True)
    path=OUT/(name+'.tif')
    with rasterio.open(path,'w',**profile) as d:
        d.write(a)
        if valid is not None:d.write_mask(valid.astype('uint8')*255)
        if descriptions:
            for i,label in enumerate(descriptions,1):d.set_band_description(i,label)
        d.update_tags(source='Sentinel-2 Collection-1 L2A; Earth Search',review_status='PENDING',classifier='Spectral baseline; NOT Amazon calibrated classifier')
    return path

def mean_valid(total,count):
    out=np.full(total.shape,NODATA,np.float32)
    np.divide(total,count,out=out,where=count>=5)
    return out

def component_masks(mask,valid):
    labels,n=ndi.label(mask&valid,np.ones((3,3)))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    largest=(labels==sizes.argmax()) if n else np.zeros(mask.shape,bool)
    excluded=mask&valid&~largest
    return largest,excluded,n

def accumulate(records):
    count=np.zeros((HEIGHT,WIDTH),np.uint16);water=np.zeros_like(count);sclwater=np.zeros_like(count);total=np.zeros((HEIGHT,WIDTH),np.float64)
    for r in records:
        if r['status']!='downloaded':continue
        with np.load(ROOT/'cache'/f"{r['date']}.npz") as z:
            v=z['valid'];count+=v;water+=z['water']&v;sclwater+=(z['scl']==6)&v;total+=np.where(v,z['mndwi'],0)
    return count,mean_valid(water,count),mean_valid(sclwater,count),mean_valid(total,count)

def rgb_composite(records):
    bands=[]
    for band in ['red','green','blue']:
        stack=[]
        for r in records:
            if r['status']!='downloaded':continue
            with np.load(ROOT/'cache'/f"{r['date']}.npz") as z:stack.append(np.where(z['valid'],z[band],np.nan))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);bands.append(np.nanmedian(np.stack(stack),axis=0))
    return np.stack(bands)

def display(rgb):return np.nan_to_num(np.clip(rgb/.25,0,1)**.65)

def annotation_layers():
    path=OUT/'review_annotations.gpkg'
    if path.exists():return # Never overwrite user annotations on regeneration.
    l,b,r,t=rasterio.windows.bounds(FOCUS,TRANSFORM)
    g=gpd.GeoDataFrame([dict(tile_id='waal_easy_01',status='pending',baseline='water_mask_occ75_all.tif',start_date='2024-01-01',end_date='2025-12-31',reviewer='',review_date='',notes='Choose baseline before accepting; unedited pixels become labels only inside accepted review areas.')],geometry=[box(l,b,r,t)],crs='EPSG:32631')
    g.to_file(path,layer='review_tiles',driver='GPKG',dataset_options={'VERSION':'1.2'})
    edits=gpd.GeoDataFrame({k:pd.Series(dtype='str') for k in ['tile_id','action','reason','confidence','notes']},geometry=gpd.GeoSeries([],crs='EPSG:32631'),crs='EPSG:32631')
    edits.to_file(path,layer='manual_edits',driver='GPKG',geometry_type='Polygon')
    ignored=gpd.GeoDataFrame({'reason':pd.Series(dtype='str')},geometry=gpd.GeoSeries([],crs='EPSG:32631'),crs='EPSG:32631')
    ignored.to_file(path,layer='uncertain_areas',driver='GPKG',geometry_type='Polygon')

def main():
    inv=json.loads((ROOT/'inventory.json').read_text())
    if inv['errors']:raise RuntimeError('Resolve acquisition failures before building a complete composite')
    records=inv['results'];expected=len(json.loads((ROOT/'catalog.json').read_text()))
    if len(records)!=expected:raise RuntimeError(f'Acquisition still incomplete: {len(records)}/{expected}')
    count,occ,sclocc,mean=accumulate(records);valid=count>=5;mask=(occ>=.75)&valid
    largest,excluded,n=component_masks(mask,valid)
    write('valid_observation_count',count,nodata=None)
    write('spectral_water_occurrence',occ);write('scl_water_occurrence',sclocc);write('mean_mndwi_NOT_probability',mean)
    for name,data in [('water_mask_occ75_all',mask),('water_mask_occ75_largest_component',largest),('excluded_components_retained',excluded)]:write(name,np.where(valid,data.astype('uint8'),255).astype('uint8'),nodata=255)
    write('excluded_component_occurrence',np.where(excluded,occ,NODATA).astype('float32'))
    rgb=rgb_composite(records);write('rgb_median_reflectance',np.where(valid[None],rgb,NODATA).astype('float32'),descriptions=['Red reflectance','Green reflectance','Blue reflectance'])
    vis=display(rgb);write('rgb_median_display',(vis*255).astype('uint8'),nodata=None,valid=valid,descriptions=['Red','Green','Blue'])
    annual={}
    for year in ['2024','2025']:
        rr=[r for r in records if r['date'].startswith(year)];c,o,s,m=accumulate(rr)
        write(f'spectral_water_occurrence_{year}',o);write(f'valid_observation_count_{year}',c,nodata=None)
        annual[year]=dict(dates=len(rr),dates_with_valid_pixels=sum(r['valid_pixels']>0 for r in rr),count_min=int(c.min()),count_median=float(np.median(c)),count_max=int(c.max()))
        best=max([r for r in rr if r['status']=='downloaded' and r['date'][5:7] in ['05','06','07','08','09']],key=lambda r:r['valid_pixels'])
        with np.load(ROOT/'cache'/f"{best['date']}.npz") as z:
            v=z['valid'];a=np.stack([z[b] for b in ['red','green','blue']]);write('dated_RGB_'+best['date'],(display(a)*255).astype('uint8'),nodata=None,valid=v,descriptions=['Red','Green','Blue'])
        annual[year]['dated_reference']=best['date']
    annotation_layers()
    sl=np.s_[int(FOCUS.row_off):int(FOCUS.row_off+FOCUS.height),int(FOCUS.col_off):int(FOCUS.col_off+FOCUS.width)]
    fig,axs=plt.subplots(2,2,figsize=(14,12))
    for ax,a,title,cmap in zip(axs.ravel(),[np.moveaxis(vis,0,-1),np.where(valid,occ,np.nan),mask,largest],['2024–2025 clear-observation median RGB','Spectral-rule water occurrence · not calibrated probability','Occurrence ≥ 75% · all components','Largest component · excluded water retained separately'],[None,'Blues','gray','gray']):
        im=ax.imshow(a,cmap=cmap,vmin=0,vmax=1);ax.set_title(title,fontsize=10);ax.axis('off')
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((FOCUS.col_off,FOCUS.row_off),FOCUS.width,FOCUS.height,fill=False,edgecolor='#ffb000',linewidth=1.5))
        if cmap=='Blues':fig.colorbar(im,ax=ax,fraction=.03)
    fig.suptitle('Waal near Druten / Dodewaard · 10 m grid · Review pending',fontsize=15);fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(14,12))
    for ax,a,title in zip(axs,[np.moveaxis(vis[:,sl[0],sl[1]],0,-1),mask[sl],largest[sl]],['Review tile · median RGB · 2024–2025','Spectral occurrence ≥ 75% · unfiltered','Largest connected water component · candidate only']):
        ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'review_tile.png',dpi=180);plt.close(fig)
    monthly={f'{year}-{month:02d}':sum(r['valid_pixels']>0 for r in records if r['date'].startswith(f'{year}-{month:02d}')) for year in ['2024','2025'] for month in range(1,13)}
    summary=dict(site='Waal near Druten / Dodewaard',dates_catalogued=len(records),dates_with_usable_pixels=sum(r['valid_pixels']>0 for r in records),annual=annual,monthly_usable_dates=monthly,grid_shape=[HEIGHT,WIDTH],resolution_m=10,effective_swir_resolution_m=20,count_min=int(count.min()),count_median=float(np.median(count)),count_max=int(count.max()),pixels_below_5_observations=int((~valid).sum()),water_area_km2=float(mask.sum()/10000),largest_component_area_km2=float(largest.sum()/10000),excluded_area_km2=float(excluded.sum()/10000),water_components=n,focus_window=[FOCUS.col_off,FOCUS.row_off,FOCUS.width,FOCUS.height],focus_water_pixels=int(mask[sl].sum()),review_status='pending',probability_model_used=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
