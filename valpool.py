"""Build a test-like validation: pools drawn only from held-out layouts, 1:1 with held-out queries."""
import numpy as np, pandas as pd, json
C='cache'; D='C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'

def build(val_q, val_l, LM, seed=7, k=20):
    """val_q[i] pairs with val_l[i].  Mimic the organisers' usage-balanced, complexity-matched sampler."""
    rng=np.random.RandomState(seed); n=len(val_l)
    comp=np.log1p(LM[val_l,0]*LM[val_l,1])+np.log1p(LM[val_l,11]+LM[val_l,17])  # size + pad complexity
    order=np.argsort(comp); rank=np.empty(n,int); rank[order]=np.arange(n)
    use=np.zeros(n,int); pools=np.zeros((n,k),int); truecol=np.zeros(n,int)
    for i in range(n):
        # nearest-in-complexity window, widened, then penalise already-heavily-used layouts
        lo,hi=max(0,rank[i]-140),min(n,rank[i]+140)
        camp=np.array([j for j in order[lo:hi] if j!=i])
        w=1.0/(1.0+use[camp])**1.5
        p=w/w.sum()
        pick=rng.choice(camp,k-1,replace=False,p=p)
        use[pick]+=1
        slot=rng.randint(k)
        row=list(pick[:slot])+[i]+list(pick[slot:])
        pools[i]=row; truecol[i]=slot
    return pools, truecol

def sinkhorn(S, iters=30, tau=None):
    """S: (nq, nl) similarity.  Returns log-domain doubly-stochastic-normalised scores."""
    L=S.astype(np.float64).copy()
    if tau: L=L/tau
    for _ in range(iters):
        L=L-np.log(np.exp(L-L.max(1,keepdims=True)).sum(1,keepdims=True))-L.max(1,keepdims=True)
        L=L-np.log(np.exp(L-L.max(0,keepdims=True)).sum(0,keepdims=True))-L.max(0,keepdims=True)
    return L
