import numpy as np, json, pandas as pd
C='cache'; D='C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'
CHANCE=sum(1.0/r for r in range(1,21))/20.0
def load():
    idx=json.load(open(C+'/index.json'))
    SI={k:i for i,k in enumerate(idx['sch'])}; LI={k:i for i,k in enumerate(idx['lay'])}
    tr=pd.read_csv(D+'/train.csv'); te=pd.read_csv(D+'/test.csv')
    cc=['cand%02d'%i for i in range(1,21)]; rr=['rel%02d'%i for i in range(1,21)]
    Q=np.array([SI[i] for i in tr['id']]); CD=np.vectorize(LI.get)(tr[cc].values); R=tr[rr].values
    T=CD[np.arange(len(CD)),R.argmax(1)]
    QT=np.array([SI[i] for i in te['id']]); CT=np.vectorize(LI.get)(te[cc].values)
    return SI,LI,tr,te,Q,CD,T,QT,CT
def mrr(S, truecol):
    """S: (n,20) scores; truecol: (n,) index of true candidate. expected RR with ties."""
    st=S[np.arange(len(S)),truecol][:,None]
    ng=(S>st+1e-12).sum(1); nt=(np.abs(S-st)<=1e-12).sum(1)
    rr=np.array([np.mean([1.0/r for r in range(g+1,g+t+1)]) for g,t in zip(ng,nt)])
    raw=rr.mean(); return raw, max(0.0,(raw-CHANCE)/(1-CHANCE))
