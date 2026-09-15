import numpy as np, json, pandas as pd
from ev import load, C, D
from valpool import build
LM=np.load(C+'/lay_meta.npy')
SI,LI,tr,te,Q,CD,T,QT,CT=load()
def comp(l): return np.log1p(LM[l,0]*LM[l,1])+np.log1p(LM[l,11]+LM[l,17])
gl=comp(np.arange(len(LM)))
def spread(pools):
    v=comp(pools); return v.std(1).mean()/gl.std()
print('OFFICIAL train pools   spread %.3f'%spread(CD))
print('OFFICIAL test  pools   spread %.3f'%spread(np.vectorize(LI.get)(te[['cand%02d'%i for i in range(1,21)]].values)))
rng=np.random.RandomState(12345); perm=rng.permutation(len(Q)); val=perm[:700]
import valpool
src=open('valpool.py').read()
for w in [60,140,250,350,700]:
    code=src.replace('rank[i]-140','rank[i]-%d'%w).replace('rank[i]+140','rank[i]+%d'%w)
    g={}; exec(code,g)
    P,tc=g['build'](Q[val],T[val],LM)
    print('  synthetic window +-%3d   spread %.3f  usage max %d'%(w,spread(P),np.bincount(P.ravel()).max()))
