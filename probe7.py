"""Dense Sinkhorn measured on TEST-LIKE pools (built from held-out layouts by my own
sampler, which I verified carries no membership signal)."""
import numpy as np
from ev import load, mrr, C
from dense import dense_sinkhorn
from valpool import build
SI,LI,tr,te,Q,CD,T,QT,CT=load()
def prep(*f):
    X=np.log1p(np.maximum(np.concatenate([np.load(C+'/'+x) for x in f],1),0))
    return (X-X.mean(0))/(X.std(0)+1e-6)
XS=prep('sch_meta.npy','sch_struct.npy'); XL=prep('lay_meta.npy','lay_struct.npy')
LM=np.load(C+'/lay_meta.npy')
rng=np.random.RandomState(12345); perm=rng.permutation(len(Q)); val=perm[:700]; fit=perm[700:]
A=np.c_[XS[Q[fit]],np.ones(len(fit))]
W=np.linalg.solve(A.T@A+3*np.eye(A.shape[1]), A.T@XL[T[fit]])
P=np.c_[XS[Q[val]],np.ones(len(val))]@W
Pn=P/np.linalg.norm(P,axis=1,keepdims=True); Ln=XL/np.linalg.norm(XL,axis=1,keepdims=True)
qi,li=Q[val],T[val]
D=Pn@Ln[li].T                                   # 700x700 dense, exactly the test situation
pools,tc=build(qi,li,LM)
idx=np.arange(700)[:,None]
print('base                     raw %.4f corr %.4f'%mrr(D[idx,pools],tc))
for tau in [0.05,0.1,0.2,0.3,0.5,1.0]:
    Z=dense_sinkhorn(D,80,tau)
    print('dense sinkhorn tau=%.2f   raw %.4f corr %.4f'%((tau,)+mrr(Z[idx,pools],tc)))
Z=dense_sinkhorn(np.zeros_like(D),80,0.2)
print('LEAK CHECK constant      raw %.4f corr %.4f'%mrr(Z[idx,pools],tc))
