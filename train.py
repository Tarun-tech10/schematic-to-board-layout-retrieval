import os, sys, time, math, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from data import load_all
from model import Duo
from ev import mrr

p=argparse.ArgumentParser()
p.add_argument('--epochs',type=int,default=60); p.add_argument('--bs',type=int,default=48)
p.add_argument('--lr',type=float,default=2e-3); p.add_argument('--wd',type=float,default=1e-4)
p.add_argument('--emb',type=int,default=256); p.add_argument('--drop',type=float,default=0.1)
p.add_argument('--poolw',type=float,default=1.0); p.add_argument('--knn',type=int,default=4); p.add_argument('--seed',type=int,default=0)
p.add_argument('--nval',type=int,default=700); p.add_argument('--tag',type=str,default='r0')
p.add_argument('--aug',type=float,default=1.0); p.add_argument('--auxw',type=float,default=1.0)
a=p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark=True
d=load_all()
S,L,SM,LM=d['S'],d['L'],d['SM'],d['LM']; Q,CD,T=d['Q'],d['CD'],d['T']
n=len(Q); rng=np.random.RandomState(12345); perm=rng.permutation(n)
val=perm[:a.nval]; fit=perm[a.nval:]
valquery=set(Q[val].tolist()); vallay=set(T[val].tolist())
print('fit %d val %d'%(len(fit),len(val)))

def getS(ix):
    x=torch.from_numpy(np.ascontiguousarray(S[ix])).to(dev,non_blocking=True)
    return x.permute(0,3,1,2).float().div_(255)
def getL(ix):
    x=torch.from_numpy(np.ascontiguousarray(L[ix])).to(dev,non_blocking=True)
    return x.permute(0,3,1,2).float().div_(255)

def aug_l(x,s):
    if s<=0: return x
    B=x.shape[0]
    f=torch.rand(B,device=x.device)<0.5; x=torch.where(f[:,None,None,None], x.flip(3), x)
    f=torch.rand(B,device=x.device)<0.5; x=torch.where(f[:,None,None,None], x.flip(2), x)
    k=torch.rand(B,device=x.device)<0.5
    if k.any(): x=torch.where(k[:,None,None,None], x.transpose(2,3), x)
    return jitter(x,s)
def jitter(x,s):
    B=x.shape[0]; dev=x.device
    sc=1.0+(torch.rand(B,device=dev)-0.5)*0.16*s
    th=torch.zeros(B,2,3,device=dev)
    th[:,0,0]=sc; th[:,1,1]=sc
    th[:,:,2]=(torch.rand(B,2,device=dev)-0.5)*0.06*s
    g=F.affine_grid(th,x.shape,align_corners=False)
    return F.grid_sample(x,g,align_corners=False,padding_mode='zeros')

model=Duo(S.shape[3],SM.shape[1],L.shape[3],LM.shape[1],emb=a.emb,drop=a.drop).to(dev)
opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.wd)
steps=a.epochs*math.ceil(len(fit)/a.bs)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,a.lr,total_steps=steps,pct_start=0.15)
scaler=torch.amp.GradScaler(dev,enabled=(dev=='cuda'))
SMg=torch.from_numpy(SM).to(dev); LMg=torch.from_numpy(LM).to(dev)

@torch.no_grad()
def embed_all(idxs, tower, arr, M, bs=64):
    model.eval(); out=[]
    for i in range(0,len(idxs),bs):
        ix=idxs[i:i+bs]
        x=torch.from_numpy(np.ascontiguousarray(arr[ix])).to(dev).permute(0,3,1,2).float().div_(255)
        with torch.amp.autocast(dev,dtype=torch.float16,enabled=(dev=='cuda')):
            e=tower(x, M[torch.from_numpy(ix).to(dev)])
        out.append(e.float().cpu())
    return torch.cat(out)

def evaluate(sel):
    qi=np.unique(Q[sel]); li=np.unique(CD[sel])
    qm={v:k for k,v in enumerate(qi)}; lm={v:k for k,v in enumerate(li)}
    EQ=embed_all(qi,model.a,S,SMg); EL=embed_all(li,model.b,L,LMg)
    qq=np.array([qm[v] for v in Q[sel]]); cc=np.vectorize(lm.get)(CD[sel])
    Sc=(EQ[qq][:,None,:]*EL[cc]).sum(-1).numpy()
    tc=np.argmax(CD[sel]==T[sel][:,None],1)
    return mrr(Sc,tc)

t0=time.time(); best=0
for ep in range(a.epochs):
    model.train(); pm=np.random.permutation(len(fit)); tot=0; nb=0
    for i in range(0,len(fit),a.bs):
        b=fit[pm[i:i+a.bs]]
        if len(b)<8: continue
        qi=Q[b]; ti=T[b]
        xs=getS(qi); xl=getL(ti)
        xs=jitter(xs,a.aug*0.6); xl=aug_l(xl,a.aug)
        with torch.amp.autocast(dev,dtype=torch.float16,enabled=(dev=='cuda')):
            gs=SMg[torch.from_numpy(qi).to(dev)]; gl=LMg[torch.from_numpy(ti).to(dev)]
            ea,pa=model.a(xs,gs,aux=True)
            eb,pb=model.b(xl,gl,aux=True)
            lg=model.scale()*ea@eb.t()
            tgt=torch.arange(len(b),device=dev)
            loss=0.5*(F.cross_entropy(lg,tgt)+F.cross_entropy(lg.t(),tgt))
            if a.auxw>0:
                loss=loss+a.auxw*0.5*(F.smooth_l1_loss(pa,gl)+F.smooth_l1_loss(pb,gs))
            if a.poolw>0:
                cand=CD[b]
                mask=(cand!=ti[:,None])
                pick=np.stack([np.random.choice(cand[r][mask[r]],a.knn,replace=False) for r in range(len(b))])
                flat=pick.reshape(-1)
                xc=aug_l(getL(flat),a.aug)
                ec=model.b(xc,LMg[torch.from_numpy(flat).to(dev)]).view(len(b),a.knn,-1)
                neg=model.scale()*(ea[:,None,:]*ec).sum(-1)
                pos=model.scale()*(ea*eb).sum(-1,keepdim=True)
                loss=loss+a.poolw*F.cross_entropy(torch.cat([pos,neg],1),
                                                  torch.zeros(len(b),dtype=torch.long,device=dev))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        tot+=float(loss); nb+=1
    if (ep+1)%5==0 or ep==a.epochs-1:
        r,c=evaluate(val)
        print('ep %3d loss %.4f  VAL raw %.4f corr %.4f  %.0fs'%(ep+1,tot/max(nb,1),r,c,time.time()-t0),flush=True)
        if c>best:
            best=c; torch.save(model.state_dict(),'ck_%s.pt'%a.tag)
    else:
        print('ep %3d loss %.4f  %.0fs'%(ep+1,tot/max(nb,1),time.time()-t0),flush=True)
print('BEST %.4f'%best)
json.dump({'best':best,'args':vars(a)},open('res_%s.json'%a.tag,'w'))
