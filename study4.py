"""Do net-like counts transfer across the two views?"""
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import prep as P
D='C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'
SPM,LPM=P.SCH_DPI/25.4,P.LAY_DPI/25.4
tr=pd.read_csv(D+'/train.csv'); cc=['cand%02d'%i for i in range(1,21)]; rr=['rel%02d'%i for i in range(1,21)]
true=tr[cc].values[np.arange(len(tr)),tr[rr].values.argmax(1)]
rng=np.random.RandomState(1); sel=rng.choice(len(tr),700,replace=False)
K=np.ones((3,3),bool)
def sf(qid):
    img=np.asarray(Image.open(D+'/schematics/%s.png'%qid).convert('RGB'))
    cls,alp=P.decompose(img,P.SCH_BG,P.SCH_PAL,P.SCH_C)
    g=(cls==0)&(alp>60)
    _,n=ndimage.label(ndimage.binary_closing(g,K),structure=K)
    r=(cls==1)&(alp>110)
    joint=ndimage.binary_dilation(g|r,K)
    _,nj=ndimage.label(joint,structure=K)
    return [n, nj, g.sum()/SPM**2]
def lf(lid):
    img=np.asarray(Image.open(D+'/train/layouts/%s.png'%lid).convert('RGB'))
    cls,alp=P.decompose(img,P.pick_bg(img),P.LAY_PAL,P.LAY_C)
    cu=((cls==0)|(cls==1)|(cls==2)|(cls==3))&(alp>90)
    _,n=ndimage.label(cu,structure=K)
    f=((cls==0)|(cls==2)|(cls==3))&(alp>90)
    _,nf=ndimage.label(f,structure=K)
    return [n, nf, cu.sum()/LPM**2]
with ThreadPoolExecutor(8) as ex:
    S=np.array(list(ex.map(sf,tr['id'].values[sel])))
    L=np.array(list(ex.map(lf,true[sel])))
sn=['wire_cc','wire+sym_cc','wire_mm2']; ln=['cu_cc','front_cc','cu_mm2']
for i,a in enumerate(sn):
    print('%-12s '%a+'  '.join('%s %.3f'%(b,np.corrcoef(np.log1p(S[:,i]),np.log1p(L[:,j]))[0,1]) for j,b in enumerate(ln)))
