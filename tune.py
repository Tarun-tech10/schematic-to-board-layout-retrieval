"""Pick the blend weight and Sinkhorn temperature on the held-out queries.
Uses test-like pools (built from held-out layouts only) so the closed 1:1 universe and the
candidate construction both match what happens at test time."""
import sys, numpy as np, torch
from data import load_all
from ev import mrr, C
from evalfull import build, emb
from valpool import build as buildpool
from dense import dense_sinkhorn

ck=sys.argv[1] if len(sys.argv)>1 else 'ck_b0.pt'
d=load_all(); Q,CD,T,SM,LM=d['Q'],d['CD'],d['T'],d['SM'],d['LM']
rng=np.random.RandomState(12345); perm=rng.permutation(len(Q)); val=perm[:700]; fit=perm[700:]
qi,li=Q[val],T[val]
m=build(ck,d,'model')
EQ=emb(m,'a',d['S'],SM,qi); EL=emb(m,'b',d['L'],LM,li,tta=True)
net=EQ@EL.T
A=np.c_[SM[Q[fit]],np.ones(len(fit))]
W=np.linalg.solve(A.T@A+3*np.eye(A.shape[1]), A.T@LM[T[fit]])
P=np.c_[SM[qi],np.ones(len(qi))]@W
P/=np.linalg.norm(P,axis=1,keepdims=True)+1e-9
Ln=LM[li]/(np.linalg.norm(LM[li],axis=1,keepdims=True)+1e-9)
ridge=P@Ln.T
pools,tc=buildpool(qi,li,np.load(C+'/lay_meta.npy'))
idx=np.arange(700)[:,None]
def sc(M): return mrr(M[idx,pools],tc)
z=lambda x:(x-x.mean())/x.std()
print('net alone      raw %.4f corr %.4f'%sc(net))
print('ridge alone    raw %.4f corr %.4f'%sc(ridge))
best=(None,-1)
for w in [0.0,0.1,0.2,0.3,0.4,0.5]:
    B=(1-w)*z(net)+w*z(ridge)
    for tau in [0,0.3,0.5,0.8,1.5]:
        M=dense_sinkhorn(B,60,tau) if tau>0 else B
        r,c=sc(M)
        if c>best[1]: best=((w,tau),c)
        print('  blend %.1f tau %.1f   raw %.4f corr %.4f'%(w,tau,r,c))
print('BEST blend/tau', best)
