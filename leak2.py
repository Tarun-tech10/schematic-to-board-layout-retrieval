"""Where does the membership leak come from?  Test scorers that use NO query information."""
import numpy as np
from ev import load, mrr, C
LM=np.load(C+'/lay_meta.npy')
SI,LI,tr,te,Q,CD,T,QT,CT=load()
tc=np.argmax(CD==T[:,None],1)
def comp(l): return np.log1p(LM[l,0]*LM[l,1])+np.log1p(LM[l,11]+LM[l,17])
for nm,v in [('board area',np.log1p(LM[:,0]*LM[:,1])),('pad complexity',np.log1p(LM[:,11]+LM[:,17])),
             ('combined',comp(np.arange(len(LM))))]:
    P=v[CD]
    med=np.median(P,1,keepdims=True); mean=P.mean(1,keepdims=True)
    print('%-15s  -|x-poolmedian|  raw %.4f corr %.4f   -|x-poolmean| raw %.4f corr %.4f'%
          ((nm,)+mrr(-np.abs(P-med),tc)+mrr(-np.abs(P-mean),tc)))
# rank of the true layout inside its pool, by complexity
P=comp(CD); r=(P<P[np.arange(len(P)),tc][:,None]).sum(1)
print('rank of true by complexity within pool: mean %.2f (uniform would be 9.5)'%r.mean())
print('histogram', np.bincount(r,minlength=20))
