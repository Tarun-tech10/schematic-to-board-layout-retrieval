import numpy as np
from ev import *
SM=np.load(C+'/sch_meta.npy'); LM=np.load(C+'/lay_meta.npy')
SI,LI,tr,te,Q,CD,T,QT,CT=load()
def feat(M):
    X=np.log1p(np.maximum(M,0))
    return np.concatenate([X, X[:,:1]*0+1],1)
XS=feat(SM); XL=feat(LM)
XS=(XS-XS.mean(0))/(XS.std(0)+1e-6); XL=(XL-XL.mean(0))/(XL.std(0)+1e-6)
rng=np.random.RandomState(0); perm=rng.permutation(len(Q)); fit=perm[:4000]; val=perm[4000:]
A=XS[Q[fit]]; B=XL[T[fit]]
lam=10.0
Wm=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]), A.T@B)
P=XS[Q[val]]@Wm                       # predicted layout feature
Cand=XL[CD[val]]                      # (n,20,d)
S=-((Cand-P[:,None,:])**2).sum(-1)
truecol=np.argmax(CD[val]==T[val][:,None],1)
print('ridge meta->meta   raw %.4f  corrected %.4f'%mrr(S,truecol))
S0=np.zeros_like(S); print('constant           raw %.4f  corrected %.4f'%mrr(S0,truecol))
# cosine on whitened residual
Pn=P/np.linalg.norm(P,axis=1,keepdims=True); Cn=Cand/np.linalg.norm(Cand,axis=-1,keepdims=True)
S2=(Cn*Pn[:,None,:]).sum(-1)
print('ridge cosine       raw %.4f  corrected %.4f'%mrr(S2,truecol))
