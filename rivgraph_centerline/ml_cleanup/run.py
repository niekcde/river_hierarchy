"""Four-fold river-held-out benchmark. No test river is used for model selection."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/river-mask-mpl')
import argparse, json, time, platform
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import torch
from core import OUT, CROP, SEED, HALO, dump, load_tile, features, cleanup, counts, scores
from models import UNet, tensor_inputs, objective
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def select_threshold(probs, records):
    trials=[]
    for t in [.2,.35,.5,.65,.8]:
        by={}
        for p,r in zip(probs,records):
            if r['stratum']!='natural':continue
            y=load_tile(r)['target'][CROP]; pred=p>=t
            a=by.setdefault(r['river'],[0,0]);a[0]+=int((pred&y).sum());a[1]+=int((pred|y).sum())
        score=float(np.mean([a/b for a,b in by.values() if b]))
        trials.append(dict(threshold=t,macro_iou=score))
    best=max(trials,key=lambda x:x['macro_iou'])
    return best['threshold'],trials

def rf_fit(records,raw,seed):
    rng=np.random.default_rng(seed);xx=[];yy=[]
    for r in records:
        z=load_tile(r);f=features(z['base'],z['raw'],r['pixel_size'],raw)[CROP].reshape(-1,13 if raw else 9)
        y=z['target'][CROP].ravel();b=z['base'][CROP].ravel()
        idx=rng.choice(len(y),384,replace=False);changed=np.flatnonzero(y!=b)
        if len(changed):idx=np.unique(np.r_[idx,rng.choice(changed,min(128,len(changed)),replace=False)])
        xx.append(f[idx]);yy.append(y[idx])
    model=ExtraTreesClassifier(n_estimators=64,max_depth=20,min_samples_leaf=5,max_features=.8,n_jobs=4,random_state=seed)
    model.fit(np.concatenate(xx),np.concatenate(yy));return model

def rf_predict(model,records,raw):
    result=[]
    for r in records:
        z=load_tile(r);f=features(z['base'],z['raw'],r['pixel_size'],raw)[CROP]
        result.append(model.predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(f.shape[:2]).astype('float32'))
    return result

def cnn_predict(model,records):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(records),8):
            x=torch.from_numpy(np.stack([tensor_inputs(load_tile(r)) for r in records[start:start+8]]))
            result.extend(model(x.to(DEVICE)).sigmoid().cpu().numpy()[:,0,HALO:-HALO,HALO:-HALO])
    return result

def cnn_fit(train,val,folder,epochs,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=UNet().to(DEVICE);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    cache=[load_tile(r) for r in train]
    probs=np.array([.5/40 if r['stratum']=='natural' else .5/16 for r in train]);probs/=probs.sum()
    # Fixed validation subset for checkpoint selection, full validation set for threshold.
    vr=[]
    for river in sorted({r['river'] for r in val}):
        for stratum,n in [('natural',4),('edit_add',2),('edit_remove',2)]:
            vr += [r for r in val if r['river']==river and r['stratum']==stratum][:n]
    best=-1;history=[];stale=0
    for epoch in range(epochs):
        start=time.time();model.train();losses=[]
        idx=rng.choice(len(train),192,p=probs,replace=True)
        for step in range(0,len(idx),8):
            xs=[];ys=[]
            for k in idx[step:step+8]:
                z=cache[k];x=tensor_inputs(z);y=z['target'][None].astype('float32')
                rot=int(rng.integers(4));x=np.rot90(x,rot,axes=(-2,-1));y=np.rot90(y,rot,axes=(-2,-1))
                if rng.random()<.5:x=x[...,::-1];y=y[...,::-1]
                xs.append(x.copy());ys.append(y.copy())
            x=torch.from_numpy(np.stack(xs)).to(DEVICE);y=torch.from_numpy(np.stack(ys)).to(DEVICE)
            optimizer.zero_grad(set_to_none=True);loss=objective(model(x),y,x[:,0:1]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();losses.append(float(loss.detach()))
        pred=cnn_predict(model,vr);t,trials=select_threshold(pred,vr);score=max(v['macro_iou'] for v in trials)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_macro_iou=score,seconds=time.time()-start))
        print('CNN',folder.name,'epoch',epoch+1,'loss',round(history[-1]['loss'],4),'val IoU',round(score,4),'sec',round(time.time()-start,1),flush=True)
        if score>best+1e-4:
            best=score;stale=0;torch.save(model.state_dict(),folder/'unet.pt')
        else:stale+=1
        dump(folder/'training.json',history)
        if stale>=6:break
    model.load_state_dict(torch.load(folder/'unet.pt',weights_only=True,map_location=DEVICE));return model

def evaluate(method,probs,recs,threshold,fold,folder,rows):
    for p,r in zip(probs,recs):
        z=load_tile(r);b=z['base'][CROP];y=z['target'][CROP]
        if method=='unet_conservative':pred=np.where(b,p>=.1,p>=.9)
        else:pred=p>=threshold
        c=counts(pred,y,b,r['pixel_size'])
        rows.append(dict(fold=fold,method=method,tile=r['file'],river=r['river'],stratum=r['stratum'],threshold=threshold,**c))
        path=folder/'predictions'/method/(Path(r['file']).stem+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,probability=np.asarray(p,dtype='float16'),prediction=pred)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int);ap.add_argument('--epochs',type=int,default=18);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rec=json.loads((OUT/'tiles.json').read_text());folds=json.loads((OUT/'splits.json').read_text())
    dump(OUT/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,seed=SEED,threads=4,device=str(DEVICE)))
    for split in folds:
        i=split['fold']
        if args.fold is not None and i!=args.fold:continue
        folder=OUT/f'fold_{i}';folder.mkdir(exist_ok=True)
        if (folder/'DONE.json').exists():print('Skipping completed',i,flush=True);continue
        train=[r for r in rec if r['river'] in split['train']];val=[r for r in rec if r['river'] in split['validation']];test=[r for r in rec if r['river'] in split['test']]
        assert not(set(split['train'])&set(split['test']) or set(split['validation'])&set(split['test']) or set(split['train'])&set(split['validation']))
        rows=json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else []
        selection=json.loads((folder/'selection.json').read_text()) if (folder/'selection.json').exists() else {}
        for method in ['unchanged','rules']:
            if sum(r['method']==method for r in rows)==len(test):continue
            p=[(load_tile(r)['base'] if method=='unchanged' else cleanup(load_tile(r)['base']))[CROP] for r in test]
            evaluate(method,p,test,.5,i,folder,rows)
        dump(folder/'metrics.json',rows)
        for raw in [False,True]:
            name='trees_raw' if raw else 'trees_binary'
            if sum(r['method']==name for r in rows)==len(test):continue
            modelpath=folder/(name+'.joblib')
            print('Training',i,name,flush=True)
            model=joblib.load(modelpath) if modelpath.exists() else rf_fit(train,raw,SEED+i)
            joblib.dump(model,modelpath)
            vp=rf_predict(model,val,raw);threshold,trials=select_threshold(vp,val);selection[name]=dict(threshold=threshold,validation=trials)
            p=rf_predict(model,test,raw);evaluate(name,p,test,threshold,i,folder,rows)
            dump(folder/'metrics.json',rows);dump(folder/'selection.json',selection)
        print('Training',i,'unet',flush=True)
        model=cnn_fit(train,val,folder,args.epochs,SEED+i)
        vp=cnn_predict(model,val);threshold,trials=select_threshold(vp,val);selection['unet']=dict(threshold=threshold,validation=trials)
        p=cnn_predict(model,test);evaluate('unet',p,test,threshold,i,folder,rows)
        evaluate('unet_conservative',p,test,.5,i,folder,rows)
        dump(folder/'selection.json',selection);dump(folder/'metrics.json',rows)
        dump(folder/'DONE.json',dict(fold=i,rows=len(rows),split=split))
        print('DONE fold',i,flush=True)

if __name__=='__main__':main()
