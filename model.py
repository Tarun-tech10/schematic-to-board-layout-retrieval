import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class Tower(nn.Module):
    def __init__(s, cin, nmeta, widths=(32,64,96,128,192,256), emb=256, drop=0.1):
        super().__init__()
        L=[]; c=cin
        for i,w in enumerate(widths):
            L += [nn.Conv2d(c,w,3,2 if i<5 else 1,1,bias=False), nn.BatchNorm2d(w), nn.ReLU(True),
                  nn.Conv2d(w,w,3,1,1,bias=False), nn.BatchNorm2d(w), nn.ReLU(True)]
            c=w
        s.body=nn.Sequential(*L); s.drop=nn.Dropout(drop)
        s.mlp=nn.Sequential(nn.Linear(nmeta,128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Linear(128,128), nn.ReLU(True))
        s.head=nn.Sequential(nn.Linear(2*c+128,512), nn.BatchNorm1d(512), nn.ReLU(True), nn.Linear(512,emb))
    def forward(s,x,m):
        h=s.body(x)
        g=torch.cat([h.mean((2,3)), h.amax((2,3)), s.mlp(m)],1)
        return F.normalize(s.head(s.drop(g)),dim=1)

class Duo(nn.Module):
    def __init__(s, cs, ms, cl, ml, emb=256, drop=0.1):
        super().__init__()
        s.a=Tower(cs,ms,emb=emb,drop=drop); s.b=Tower(cl,ml,emb=emb,drop=drop)
        s.logit_scale=nn.Parameter(torch.tensor(np.log(1/0.07),dtype=torch.float32))
    def scale(s): return s.logit_scale.exp().clamp(max=100.0)
