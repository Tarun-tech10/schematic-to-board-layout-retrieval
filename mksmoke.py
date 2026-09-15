import os, shutil, pandas as pd, numpy as np
D='C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'; O='smoke'
cc=['cand%02d'%i for i in range(1,21)]
tr=pd.read_csv(D+'/train.csv').head(60); te=pd.read_csv(D+'/test.csv').head(20)
for sub in ['schematics','train/layouts','test/layouts']: os.makedirs(O+'/'+sub,exist_ok=True)
tr.to_csv(O+'/train.csv',index=False); te.to_csv(O+'/test.csv',index=False)
for i in list(tr['id'])+list(te['id']): shutil.copy(D+'/schematics/%s.png'%i, O+'/schematics/')
for v in set(tr[cc].values.ravel()): shutil.copy(D+'/train/layouts/%s.png'%v, O+'/train/layouts/')
for v in set(te[cc].values.ravel()): shutil.copy(D+'/test/layouts/%s.png'%v, O+'/test/layouts/')
print('smoke set: %d sch, %d train lay, %d test lay'%(len(tr)+len(te),len(set(tr[cc].values.ravel())),len(set(te[cc].values.ravel()))))
