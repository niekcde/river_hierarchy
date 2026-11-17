"""Small residual U-Net; weighted bank/edit supervision and soft connectivity loss."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import HALO

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())
    def forward(self,x):return self.net(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=Block(2,8);self.b=Block(8,16);self.c=Block(16,32);self.d=Block(32,64)
        self.u=Block(96,32);self.v=Block(48,16);self.w=Block(24,8);self.out=nn.Conv2d(8,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.max_pool2d(a,2));c=self.c(F.max_pool2d(b,2));d=self.d(F.max_pool2d(c,2))
        u=self.u(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        v=self.v(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        w=self.w(torch.cat([F.interpolate(v,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(w)+4*(x[:,0:1]-.5)

def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))

def soft_skeleton(x,iterations=4):
    opened=F.max_pool2d(soft_erode(x),3,1,1);s=F.relu(x-opened)
    for _ in range(iterations):
        x=soft_erode(x);d=F.relu(x-F.max_pool2d(soft_erode(x),3,1,1));s=s+F.relu(d-s*d)
    return s

def objective(logits,y,b):
    sl=np.s_[:,:,HALO:-HALO,HALO:-HALO]
    logits=logits[sl];y=y[sl];b=b[sl];p=logits.sigmoid()
    boundary=F.max_pool2d(y,3,1,1)+F.max_pool2d(-y,3,1,1)
    weight=1+3*boundary+3*(y!=b).float()
    bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*weight).mean()
    dims=(1,2,3);dice=1-((2*(p*y).sum(dims)+1)/(p.sum(dims)+y.sum(dims)+1)).mean()
    sp=soft_skeleton(p);sy=soft_skeleton(y)
    precision=((sp*y).sum(dims)+1)/(sp.sum(dims)+1)
    recall=((sy*p).sum(dims)+1)/(sy.sum(dims)+1)
    cl=1-(2*precision*recall/(precision+recall+1e-6)).mean()
    return bce+.5*dice+.2*cl

def tensor_inputs(z):return np.stack([z['base'].astype(np.float32),z['raw']],axis=0)
