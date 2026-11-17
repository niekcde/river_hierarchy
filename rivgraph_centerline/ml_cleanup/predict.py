"""Apply a trained U-Net to an original/prepared pair, never reading cleaned labels."""
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import torch
from core import CORE,HALO,SIDE
from models import UNet

def predict(model_path,prepared,source,source_threshold,output,decision_threshold=.5):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_name(output.stem+'_probability.tif').exists():raise FileExistsError('Refusing to overwrite an existing prediction')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model=UNet().to(device);model.load_state_dict(torch.load(model_path,weights_only=True,map_location=device));model.eval();torch.set_num_threads(4)
    with rasterio.open(prepared) as b,rasterio.open(source) as s:
        profile=b.profile.copy();profile.update(dtype='uint8',nodata=255,compress='deflate',tiled=True,blockxsize=256,blockysize=256)
        probprofile=profile.copy();probprofile.update(dtype='float32',nodata=-1)
        with WarpedVRT(s,crs=b.crs,transform=b.transform,width=b.width,height=b.height,resampling=Resampling.bilinear,src_nodata=None,nodata=0) as raw,rasterio.open(output,'w',**profile) as dst,rasterio.open(output.with_name(output.stem+'_probability.tif'),'w',**probprofile) as pdst:
            windows=[Window(c,r,min(CORE,b.width-c),min(CORE,b.height-r)) for r in range(0,b.height,CORE) for c in range(0,b.width,CORE)]
            for i in range(0,len(windows),8):
                xs=[]
                for w in windows[i:i+8]:
                    # VRT disallows boundless reads; pad the observed edge by replication.
                    r,c=int(w.row_off),int(w.col_off)
                    r0=max(0,r-HALO);c0=max(0,c-HALO);r1=min(b.height,r+CORE+HALO);c1=min(b.width,c+CORE+HALO)
                    iw=Window(c0,r0,c1-c0,r1-r0)
                    base=b.read(1,window=iw)>0;v=raw.read(1,window=iw,out_dtype='float32');v=np.clip(v/source_threshold,0,5)/5
                    pad=((r0-(r-HALO),r+CORE+HALO-r1),(c0-(c-HALO),c+CORE+HALO-c1))
                    xs.append(np.stack([np.pad(base,pad,mode='edge'),np.pad(v,pad,mode='edge')]).astype('float32'))
                with torch.no_grad():p=model(torch.from_numpy(np.stack(xs)).to(device)).sigmoid().cpu().numpy()[:,0]
                for v,w in zip(p,windows[i:i+8]):
                    a=v[HALO:HALO+int(w.height),HALO:HALO+int(w.width)]
                    dst.write((a>=decision_threshold).astype('uint8'),1,window=w);pdst.write(a,1,window=w)
                if i%80==0:print(i,'/',len(windows),'tiles',flush=True)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--prepared',required=True);p.add_argument('--source',required=True);p.add_argument('--source-threshold',type=float,required=True);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--output',required=True);a=p.parse_args()
    if a.source_threshold<=0:raise ValueError('Source threshold must be positive')
    predict(a.model,a.prepared,a.source,a.source_threshold,a.output,a.threshold)
