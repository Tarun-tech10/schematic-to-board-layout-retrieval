import numpy as np
from ev import load, mrr, C
from sink import pool_sinkhorn
from valpool import build
LM=np.load(C+'/lay_meta.npy')
SI,LI,tr,te,Q,CD,T,QT,CT=load()
tc=np.argmax(CD==T[:,None],1)
Z=pool_sinkhorn(np.zeros_like(CD,dtype=float),CD,len(LM),iters=150,tau=1.0)
top1=(Z.argmax(1)==tc).mean()
print('OFFICIAL train pools: constant+sinkhorn top-1 %.3f (chance 0.050)   corr MRR %.4f'%((top1,)+mrr(Z,tc)[1:]))
# my own usage-balanced complexity-matched sampler, same sizes
rng=np.random.RandomState(12345); perm=rng.permutation(len(Q)); val=perm[:700]
P,t2=build(Q[val],T[val],LM)
Z2=pool_sinkhorn(np.zeros_like(P,dtype=float),P,700,iters=150,tau=1.0)
print('MY synthetic pools:   constant+sinkhorn top-1 %.3f                  corr MRR %.4f'%(((Z2.argmax(1)==t2).mean(),)+mrr(Z2,t2)[1:]))
# does the same structure appear in the TEST pools?  (no labels -- measure confidence instead)
Zt=pool_sinkhorn(np.zeros((len(CT),20)),CT,len(LM),iters=150,tau=1.0)
Zo=pool_sinkhorn(np.zeros_like(CD,dtype=float),CD,len(LM),iters=150,tau=1.0)
def conf(Z):
    p=np.exp(Z-Z.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    return p.max(1).mean()
print('mean top prob: train pools %.3f   test pools %.3f   (uniform 0.05)'%(conf(Zo),conf(Zt)))
