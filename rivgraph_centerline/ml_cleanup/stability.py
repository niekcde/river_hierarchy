"""Repeat the same frozen CNN experiment with a second training seed."""
import json
import numpy as np
import torch
from core import OUT,SEED,dump
from run import cnn_fit,cnn_predict,select_threshold,evaluate

def main():
    torch.set_num_threads(3);torch.set_num_interop_threads(1)
    rec=json.loads((OUT/'tiles.json').read_text());splits=json.loads((OUT/'splits.json').read_text())
    root=OUT/'second_seed';root.mkdir(exist_ok=True)
    for s in splits:
        i=s['fold'];folder=root/f'fold_{i}';folder.mkdir(exist_ok=True)
        if (folder/'DONE.json').exists():continue
        train=[r for r in rec if r['river'] in s['train']];val=[r for r in rec if r['river'] in s['validation']];test=[r for r in rec if r['river'] in s['test']]
        model=cnn_fit(train,val,folder,18,SEED+10000+i)
        vp=cnn_predict(model,val);threshold,trials=select_threshold(vp,val)
        p=cnn_predict(model,test);rows=[];evaluate('unet',p,test,threshold,i,folder,rows)
        dump(folder/'selection.json',dict(seed=SEED+10000+i,threshold=threshold,validation=trials));dump(folder/'metrics.json',rows);dump(folder/'DONE.json',s)
        print('SECOND SEED DONE',i,flush=True)
if __name__=='__main__':main()
