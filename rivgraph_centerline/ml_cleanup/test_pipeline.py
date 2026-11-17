import json
from types import SimpleNamespace
import numpy as np
import pytest
from affine import Affine
from core import aligned_offset, cleanup, counts, scores, features, holes, OUT, CROP, load_tile


def raster(transform,shape=(100,100),crs='EPSG:32617'):
    return SimpleNamespace(transform=transform,height=shape[0],width=shape[1],crs=crs)

def test_crop_alignment():
    t=Affine(10,0,100,0,-10,1000)
    assert aligned_offset(raster(t),raster(t*Affine.translation(5,8),(20,30)))==(5,8)

@pytest.mark.parametrize('delta',[.25,-1,100])
def test_reject_bad_offset(delta):
    t=Affine(10,0,100,0,-10,1000)
    with pytest.raises(ValueError):aligned_offset(raster(t),raster(t*Affine.translation(delta,0),(20,20)))

def test_reject_crs_and_scale():
    t=Affine(10,0,100,0,-10,1000)
    with pytest.raises(ValueError):aligned_offset(raster(t),raster(t,crs='EPSG:4326'))
    with pytest.raises(ValueError):aligned_offset(raster(t),raster(t*Affine.scale(2)))

def test_cleanup_preserves_real_island_and_channel():
    m=np.zeros((40,40),bool);m[5:35,5:35]=1;m[15:20,15:20]=0;m[10,10]=0;m[1,1]=1
    p=cleanup(m)
    assert not p[1,1] and p[10,10] and not p[17,17]
    assert p[6:34,6:14].all()

def test_no_fill_exterior_land():
    m=np.ones((20,20),bool);m[0,5]=0
    assert not cleanup(m,min_water=1)[0,5]

def test_perfect_metrics():
    m=np.zeros((40,40),bool);m[5:35,5:35]=1;m[15:20,15:20]=0
    c=counts(m,m,m,10);s=scores(c)
    for key in ['iou','dice','boundary_f1','cldice','island_iou']:assert s[key]==1
    assert c['damage']==0 and c['islands_true']==1

def test_bad_bridge_reduces_connectivity():
    y=np.zeros((40,40),bool);y[18:22,4:36]=1;p=y.copy();p[:,19:21]=0
    c=counts(p,y,y,10)
    assert c['components_pred']==2 and c['components_true']==1
    assert scores(c)['cldice']<1 and c['damage']>0

def test_edit_direction_counts():
    b=np.zeros((40,40),bool);b[10:20,10:20]=1;y=b.copy();y[10:12,10:20]=0;y[20:22,10:20]=1
    c=counts(y,y,b,10)
    assert c['removed_needed']==20 and c['added_needed']==20
    assert scores(c)['edit_precision']==1 and scores(c)['damage_rate']==0

def test_empty_metrics_serializable():
    m=np.zeros((40,40),bool);json.dumps(scores(counts(m,m,m,10)),allow_nan=False)

def test_boundary_shift_detected():
    y=np.zeros((80,80),bool);y[20:60,20:60]=1;p=np.roll(y,5,axis=1)
    assert scores(counts(p,y,y,10))['boundary_f1']<1

def test_features_independent_and_finite():
    b=np.zeros((64,64),bool);b[15:35,15:35]=1
    a=features(b,b.astype('float32'),10);c=features(b,1-b.astype('float32'),10)
    assert a.shape==(64,64,13) and np.isfinite(a).all()
    np.testing.assert_array_equal(a[:,:,:9],c[:,:,:9])
    assert not np.array_equal(a[:,:,9:],c[:,:,9:])

def test_artifact_split_integrity():
    splits=json.loads((OUT/'splits.json').read_text());rec=json.loads((OUT/'tiles.json').read_text())
    test=[]
    for f in splits:
        a,b,c=map(set,[f['train'],f['validation'],f['test']])
        assert not(a&b or a&c or b&c);test+=f['test']
    assert len(test)==len(set(test)) and set(test)=={r['river'] for r in rec}

def test_real_arrays_and_labels():
    rec=json.loads((OUT/'tiles.json').read_text())
    for r in rec[::19]:
        z=load_tile(r)
        assert z['base'].shape==z['target'].shape==z['raw'].shape==(256,256)
        assert np.isfinite(z['raw']).all() and 0<=z['raw'].min()<=z['raw'].max()<=1
        assert np.count_nonzero(z['base'][CROP]!=z['target'][CROP])==r['changed']

def test_network_gradient_and_reload(tmp_path):
    import torch
    from models import UNet,objective
    torch.set_num_threads(2);torch.manual_seed(5)
    x=torch.rand(2,2,96,96);y=(x[:,:1]>.5).float();m=UNet()
    out=m(x);assert out.shape==y.shape
    loss=objective(out,y,x[:,:1]);loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    torch.save(m.state_dict(),tmp_path/'model.pt');n=UNet();n.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True))
    torch.testing.assert_close(m(x),n(x))

def test_streaming_prediction_edges_and_georeferencing(tmp_path):
    import torch,rasterio,hashlib
    from models import UNet
    from predict import predict
    m=UNet();torch.save(m.state_dict(),tmp_path/'m.pt')
    profile=dict(driver='GTiff',height=205,width=213,count=1,dtype='uint8',crs='EPSG:32617',transform=Affine(10,0,100,0,-10,1000),nodata=0)
    a=np.zeros((205,213),np.uint8);a[20:180,50:80]=1
    for name,data in [('b.tif',a),('s.tif',a*80)]:
        with rasterio.open(tmp_path/name,'w',**profile) as f:f.write(data,1)
    before=hashlib.sha256((tmp_path/'b.tif').read_bytes()).hexdigest()
    predict(tmp_path/'m.pt',tmp_path/'b.tif',tmp_path/'s.tif',20,tmp_path/'out.tif')
    with rasterio.open(tmp_path/'out.tif') as f:
        assert f.shape==a.shape and f.transform==profile['transform'] and f.nodata==255
        assert set(np.unique(f.read(1)))<={0,1}
    with rasterio.open(tmp_path/'out_probability.tif') as f:
        p=f.read(1);assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    assert before==hashlib.sha256((tmp_path/'b.tif').read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):predict(tmp_path/'m.pt',tmp_path/'b.tif',tmp_path/'s.tif',20,tmp_path/'out.tif')

def test_selection_uses_validation_natural_only(monkeypatch):
    import run
    rec=[dict(river='a',stratum='natural'),dict(river='b',stratum='edit_add')]
    y=np.ones((256,256),bool)
    monkeypatch.setattr(run,'load_tile',lambda r:dict(target=y))
    p=[np.full((192,192),.4),np.zeros((192,192))]
    threshold,trials=run.select_threshold(p,rec)
    assert threshold in [.2,.35] and max(t['macro_iou'] for t in trials)==1

def test_all_land_prediction_does_not_get_high_accuracy_score():
    y=np.zeros((80,80),bool);y[20:60,35:45]=1
    s=scores(counts(np.zeros_like(y),y,y,10))
    assert s['iou']==0 and s['dice']==0 and s['damage_rate']>0
    assert s['boundary_f1']==0 and s['cldice']==0


def test_holes_exclude_boundary_intersections():
    y=np.ones((40,40),bool);y[0:10,10:20]=0;y[20:25,20:25]=0
    mask,count=holes(y)
    assert count==1 and mask.sum()==25

def test_gpu_trained_checkpoint_portable_to_cpu():
    import torch
    from models import UNet,tensor_inputs
    from pathlib import Path
    folder=OUT/'fold_0'
    if not(folder/'DONE.json').exists():pytest.skip('Training has not completed')
    m=UNet();m.load_state_dict(torch.load(folder/'unet.pt',weights_only=True,map_location='cpu'));m.eval()
    split=json.loads((OUT/'splits.json').read_text())[0]
    rec=next(r for r in json.loads((OUT/'tiles.json').read_text()) if r['river'] in split['test'])
    with torch.no_grad():p=m(torch.from_numpy(tensor_inputs(load_tile(rec))[None])).sigmoid().numpy()[0,0][CROP]
    with np.load(folder/'predictions'/'unet'/(Path(rec['file']).stem+'.npz')) as f:
        np.testing.assert_allclose(p,f['probability'],atol=.002,rtol=.002)
