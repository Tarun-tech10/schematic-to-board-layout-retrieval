"""Clean competition correction: Sinkhorn on the DENSE query x layout score matrix.
Uses only model scores plus the brief's stated 1:1 property -- never pool membership."""
import numpy as np

def dense_sinkhorn(S, iters=60, tau=1.0):
    L=S.astype(np.float64)/tau
    for _ in range(iters):
        L=L-(L.max(1,keepdims=True)+np.log(np.exp(L-L.max(1,keepdims=True)).sum(1,keepdims=True)))
        L=L-(L.max(0,keepdims=True)+np.log(np.exp(L-L.max(0,keepdims=True)).sum(0,keepdims=True)))
    return L
