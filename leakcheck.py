"""Does pool MEMBERSHIP alone carry the answer?  Run the same normalisation on constant scores."""
import numpy as np
from ev import load, mrr, C
from sink import pool_sinkhorn
LM=np.load(C+'/lay_meta.npy')
SI,LI,tr,te,Q,CD,T,QT,CT=load()
tc=np.argmax(CD==T[:,None],1)
S0=np.zeros_like(CD,dtype=float)
print('constant scores, no normalisation      raw %.4f corr %.4f'%mrr(S0,tc))
for tau,it in [(0.3,40),(1.0,150),(5.0,150),(20.0,150)]:
    Z=pool_sinkhorn(S0,CD,len(LM),iters=it,tau=tau)
    print('constant + sinkhorn tau=%5.1f it=%3d   raw %.4f corr %.4f'%((tau,it)+mrr(Z,tc)))
# raw usage count as a score
u=np.bincount(CD.ravel(),minlength=len(LM)).astype(float)
print('usage count distribution: min %d max %d'%(u[u>0].min(),u.max()))
print('score = -usage                         raw %.4f corr %.4f'%mrr(-u[CD],tc))
print('score = +usage                         raw %.4f corr %.4f'%mrr(u[CD],tc))
