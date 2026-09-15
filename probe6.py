import numpy as np
from ev import load, mrr, C
SI,LI,tr,te,Q,CD,T,QT,CT=load()
def prep(*M):
    X=np.log1p(np.maximum(np.concatenate(M,1),0)); return (X-X.mean(0))/(X.std(0)+1e-6)
sets={
 'meta only'      : (prep(np.load(C+'/sch_meta.npy')), prep(np.load(C+'/lay_meta.npy'))),
 'struct only'    : (prep(np.load(C+'/sch_struct.npy')), prep(np.load(C+'/lay_struct.npy'))),
 'meta+struct'    : (prep(np.load(C+'/sch_meta.npy'),np.load(C+'/sch_struct.npy')),
                     prep(np.load(C+'/lay_meta.npy'),np.load(C+'/lay_struct.npy'))),
}
tc=np.argmax(CD==T[:,None],1); n=len(Q)
rng=np.random.RandomState(0); fold=rng.permutation(n)%5
for nm,(XS,XL) in sets.items():
    Ln=XL/np.linalg.norm(XL,axis=1,keepdims=True)
    best=None
    for lam in [3,10,30,100]:
        P=np.zeros((n,XL.shape[1]))
        for f in range(5):
            fit=fold!=f
            A=np.c_[XS[Q[fit]],np.ones(fit.sum())]
            W=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]), A.T@XL[T[fit]])
            P[~fit]=np.c_[XS[Q[~fit]],np.ones((~fit).sum())]@W
        Pn=P/np.linalg.norm(P,axis=1,keepdims=True)
        r,c=mrr((Pn[:,None,:]*Ln[CD]).sum(-1),tc)
        if best is None or c>best[1]: best=(lam,c,r)
    print('%-13s dims %3d+%3d   best lam %3d   raw %.4f corr %.4f'%(nm,XS.shape[1],XL.shape[1],best[0],best[2],best[1]))
