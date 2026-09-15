"""Pool-restricted Sinkhorn: exploit that the candidate set is a perfect 1:1 matching."""
import numpy as np

def pool_sinkhorn(S, pools, nl, iters=40, tau=1.0, alpha=1.0):
    """S: (nq,k) pool scores.  pools: (nq,k) layout ids in [0,nl).
    Returns competition-corrected pool scores of the same shape."""
    nq, k = S.shape
    L = S / tau
    base = L.copy()
    for _ in range(iters):
        L = L - _lse_rows(L)[:, None]                       # each query sums to 1 over its pool
        col = np.full(nl, -np.inf)
        for j in range(k):                                  # each layout sums to 1 over the pools it sits in
            col = np.logaddexp(col, _scatter_lse(pools[:, j], L[:, j], nl))
        L = L - col[pools]
    return alpha * L + (1 - alpha) * base

def _lse_rows(L):
    m = L.max(1)
    return m + np.log(np.exp(L - m[:, None]).sum(1))

def _scatter_lse(idx, val, n):
    m = np.full(n, -np.inf)
    np.maximum.at(m, idx, val)
    fin = np.isfinite(m)
    acc = np.zeros(n)
    np.add.at(acc, idx, np.exp(val - np.where(fin, m, 0.0)[idx]))
    out = np.full(n, -np.inf)
    out[fin] = m[fin] + np.log(acc[fin])
    return out
