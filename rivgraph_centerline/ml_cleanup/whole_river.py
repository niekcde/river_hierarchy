"""One complete held-out-river application, with full-extent topology diagnostics."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import ROOT,OUT,CORE,HALO,features,aligned_offset,cleanup,counts,scores,dump
from predict import predict


def tree_predict(modelpath,prepared,source,source_threshold,output,threshold):
    model=joblib.load(modelpath)
    with rasterio.open(prepared) as b,rasterio.open(source) as s:
        profile=b.profile.copy();profile.update(dtype='uint8',nodata=255,compress='deflate',tiled=True)
        with WarpedVRT(s,crs=b.crs,transform=b.transform,width=b.width,height=b.height,resampling=Resampling.bilinear,src_nodata=None,nodata=0) as raw,rasterio.open(output,'w',**profile) as dst:
            for row in range(0,b.height,CORE):
                for col in range(0,b.width,CORE):
                    h=min(CORE,b.height-row);w=min(CORE,b.width-col)
                    r0=max(0,row-HALO);c0=max(0,col-HALO);r1=min(b.height,row+CORE+HALO);c1=min(b.width,col+CORE+HALO)
                    iw=Window(c0,r0,c1-c0,r1-r0)
                    base=b.read(1,window=iw)>0;v=np.clip(raw.read(1,window=iw,out_dtype='float32')/source_threshold,0,5)/5
                    pad=((r0-(row-HALO),row+CORE+HALO-r1),(c0-(col-HALO),col+CORE+HALO-c1))
                    base=np.pad(base,pad,mode='edge');v=np.pad(v,pad,mode='edge')
                    f=features(base,v,abs(b.res[0]))[HALO:HALO+h,HALO:HALO+w]
                    p=model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(h,w)>=threshold
                    dst.write(p.astype('uint8'),1,window=Window(col,row,w,h))
                print('trees row',row,'/',b.height,flush=True)

def main():
    n='03';fold=next(f for f in json.loads((OUT/'splits.json').read_text()) if n in f['test'])
    folder=OUT/f"fold_{fold['fold']}"
    if not(folder/'DONE.json').exists():raise RuntimeError('Held-out fold not complete')
    assert n not in fold['train']+fold['validation']
    d=ROOT/n;meta=json.loads((d/'prepare_summary.json').read_text());selection=json.loads((folder/'selection.json').read_text())
    prepared=d/'masks_prepared'/f'{n}_binary_projected.tif';clean=d/'masks_cleaned'/f'{n}_cleaned.tif'
    dest=OUT/'whole_river_03';dest.mkdir(exist_ok=True)
    paths={m:dest/f'03_{m}_experimental.tif' for m in ['unet','trees_raw']}
    if not paths['unet'].exists():predict(folder/'unet.pt',prepared,meta['source_mask'],meta['threshold'],paths['unet'],selection['unet']['threshold'])
    if not paths['trees_raw'].exists():tree_predict(folder/'trees_raw.joblib',prepared,meta['source_mask'],meta['threshold'],paths['trees_raw'],selection['trees_raw']['threshold'])
    with rasterio.open(prepared) as b,rasterio.open(clean) as c:
        col,row=aligned_offset(b,c);w=Window(col,row,c.width,c.height)
        base=b.read(1,window=w)>0;y=c.read(1)>0;resolution=abs(b.res[0])
        full=b.read(1)>0;lab,num=ndi.label(full,np.ones((3,3)));sz=np.bincount(lab.ravel());sz[0]=0
        largest_full=cleanup(lab==int(sz.argmax()),min_water=1)
        largest=largest_full[row:row+c.height,col:col+c.width]
        global_path=dest/'03_global_component_experimental.tif'
        if not global_path.exists():
            profile=b.profile.copy();profile.update(dtype='uint8',nodata=255,compress='deflate')
            with rasterio.open(global_path,'w',**profile) as dst:dst.write(largest_full.astype('uint8'),1)
        del largest_full
        del full,lab
    arrays={'unchanged':base,'rules':cleanup(base),'global_largest_component':largest}
    for name,path in paths.items():
        with rasterio.open(path) as p:arrays[name]=p.read(1,window=w)>0
    result=[]
    rr=(np.arange(y.shape[0])+row)%CORE;ccols=(np.arange(y.shape[1])+col)%CORE
    seam=(np.minimum(rr,CORE-rr)[:,None]<=2)|(np.minimum(ccols,CORE-ccols)[None,:]<=2)
    for name,p in arrays.items():
        print('scoring full river',name,flush=True)
        cc=counts(p,y,base,resolution)
        lab,num=ndi.label(p,np.ones((3,3)));sz=np.bincount(lab.ravel());sz[0]=0
        result.append(dict(method=name,**scores(cc),components=int(num),target_components=cc['components_true'],largest_component_water_fraction=float(sz.max()/p.sum()) if p.any() else None,islands_pred=cc['islands_pred'],islands_true=cc['islands_true'],
             seam_error_rate=float((p[seam]!=y[seam]).mean()),interior_error_rate=float((p[~seam]!=y[~seam]).mean())))
    dump(dest/'metrics.json',dict(river=n,fold=fold,scope='Entire cleaned extent; predictions generated over entire prepared raster without target access',thresholds=selection,metrics=result,
          model_hashes={m:hashlib.sha256((folder/('unet.pt' if m=='unet' else 'trees_raw.joblib')).read_bytes()).hexdigest() for m in paths}))
    fig,axes=plt.subplots(5,1,figsize=(16,19))
    for ax,(name,a) in zip(axes,[('Prepared mask',base),('Manual target',y),('Global largest-component baseline',largest),('Held-out original-valued image trees',arrays['trees_raw']),('Held-out U-Net',arrays['unet'])]):
        ax.imshow(a[::3,::3],cmap='gray',vmin=0,vmax=1);ax.set_title(name);ax.axis('off')
    fig.tight_layout();fig.savefig(dest/'overview.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
