import numpy as np, json, pandas as pd
C='cache'; D='C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'

def metafeat(M, kind):
    X=np.log1p(np.maximum(M,0.0)).astype(np.float32)
    return X

def load_all():
    idx=json.load(open(C+'/index.json'))
    SI={k:i for i,k in enumerate(idx['sch'])}; LI={k:i for i,k in enumerate(idx['lay'])}
    S=np.load(C+'/sch.npy',mmap_mode='r'); L=np.load(C+'/lay.npy',mmap_mode='r')
    SMr=np.load(C+'/sch_meta.npy'); LMr=np.load(C+'/lay_meta.npy')
    SM=metafeat(SMr,'s'); LM=metafeat(LMr,'l')
    SM=(SM-SM.mean(0))/(SM.std(0)+1e-6); LM=(LM-LM.mean(0))/(LM.std(0)+1e-6)
    tr=pd.read_csv(D+'/train.csv'); te=pd.read_csv(D+'/test.csv')
    cc=['cand%02d'%i for i in range(1,21)]; rr=['rel%02d'%i for i in range(1,21)]
    Q=np.array([SI[i] for i in tr['id']]); CD=np.vectorize(LI.get)(tr[cc].values)
    T=CD[np.arange(len(CD)),tr[rr].values.argmax(1)]
    QT=np.array([SI[i] for i in te['id']]); CT=np.vectorize(LI.get)(te[cc].values)
    return dict(S=S,L=L,SM=SM.astype(np.float32),LM=LM.astype(np.float32),
                Q=Q,CD=CD,T=T,QT=QT,CT=CT,tr=tr,te=te)
