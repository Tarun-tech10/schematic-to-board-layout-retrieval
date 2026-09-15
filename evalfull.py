"""Score a checkpoint on the held-out queries (official pools) and dump scores for blending."""
import sys, numpy as np, torch, importlib
from data import load_all
from ev import mrr

def build(ck, d, arch, dev='cuda'):
    M=importlib.import_module(arch)
    m=M.Duo(d['S'].shape[3],d['SM'].shape[1],d['L'].shape[3],d['LM'].shape[1]).to(dev)
    m.load_state_dict(torch.load(ck,map_location=dev)); m.eval(); return m

def emb(m,tower_name,arr,M,ix,dev='cuda',tta=False):
    tower=getattr(m,tower_name); Mg=torch.from_numpy(M).to(dev); out=[]
    for i in range(0,len(ix),64):
        b=ix[i:i+64]
        x=torch.from_numpy(np.ascontiguousarray(arr[b])).to(dev).permute(0,3,1,2).float().div_(255)
        mm=Mg[torch.from_numpy(b).to(dev)]
        with torch.no_grad(), torch.amp.autocast(dev,dtype=torch.float16):
            e=tower(x,mm).float()
            if tta:
                for t in (x.flip(3),x.flip(2),x.flip(2).flip(3)):
                    e=e+tower(t,mm).float()
        out.append(torch.nn.functional.normalize(e,dim=1).cpu())
    return torch.cat(out).numpy()

def scores(ck, d, arch='model', tta=True, sel=None):
    Q,CD,T=d['Q'],d['CD'],d['T']
    m=build(ck,d,arch)
    qi=Q[sel]; allc=np.unique(CD[sel]); pos={v:k for k,v in enumerate(allc)}
    EQ=emb(m,'a',d['S'],d['SM'],qi)
    EL=emb(m,'b',d['L'],d['LM'],allc,tta=tta)
    col=np.vectorize(pos.get)(CD[sel])
    return (EQ[:,None,:]*EL[col]).sum(-1), np.argmax(CD[sel]==T[sel][:,None],1)

if __name__=='__main__':
    d=load_all()
    rng=np.random.RandomState(12345); perm=rng.permutation(len(d['Q'])); val=perm[:700]
    for ck in sys.argv[1:]:
        arch='model_v0' if 'a0' in ck else 'model'
        for tta in (False,True):
            S,tc=scores(ck,d,arch,tta,val)
            print('%-12s tta=%-5s raw %.4f corr %.4f'%(ck,tta,*mrr(S,tc)),flush=True)
        np.save(ck.replace('.pt','_val.npy'),S); np.save('val_tc.npy',tc)
