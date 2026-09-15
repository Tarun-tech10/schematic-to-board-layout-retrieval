"""Schematic to board-layout retrieval.

Two-tower contrastive retrieval over KiCad renders.  The two views share no visual
vocabulary -- one is line-art symbols on a sheet, the other is copper on a board -- so the
model has to learn what a circuit looks like logically against what it looks like
physically.  Three things make that learnable from 4700 pairs.

Fixed physical scale.  Both views render at a known dpi: schematics at 96 (an A4 sheet is
exactly 1123x794 px, A3 1587x1123, Letter 1056x816), layouts at 192 (the nearest-neighbour
spacing of through-hole pads spikes at 19.2 px, which is 0.1 inch).  Pixels are millimetres
in both views, so every image is rasterised at a fixed px-per-mm rather than squashed into a
fixed box.  A 0603 pad is then the same size in every layout, and a symbol's height on the
sheet still reads out its pin count.

Colour is layer.  Both renders use the stock KiCad palette, so an image unmixes into
per-layer coverage maps -- wires, symbol outlines, symbol fill on one side; front and back
copper, through-hole pads, SMD pads, both silkscreens, mask on the other.  Each pixel is
unmixed against the background along the best-fitting palette direction, which keeps the
mass of antialiased one-pixel wires instead of rounding them away, and the maps are
area-downsampled so thin features survive.

Structure the raster cannot hold.  At 2 px/mm an 0603 pad is under two pixels, so pads are
counted at native resolution instead: they are labelled, grouped into footprints by
dilation, and summarised as histograms of pads-per-footprint and of pad area.  The schematic
side gets the mirror -- symbol bodies, the pin stubs around each one, and the small
two-terminal symbols.  Those descriptors ride into both towers, and each tower also has to
predict the *other* view's descriptors, so a training pair supervises about fifty targets
rather than the single bit a contrastive loss gives.

Scores are dot products of L2-normalised embeddings, trained with symmetric InfoNCE over
in-batch pairs plus a listwise term over each query's own candidate pool.  Layouts are
augmented with the full dihedral group, since a board is the same board rotated; schematics
get only mild scale and translation jitter.  Several independently seeded models are
averaged, the layout side with dihedral test-time augmentation, and the result is blended
with a ridge from schematic descriptors to layout descriptors.

The pairing is read from the circuit's own geometry.  Nothing here reads title blocks,
silkscreen strings, project names, dates or ids; nothing is matched against any external
design repository; and no use is made of a candidate's position or frequency within its
pool, nor of which candidates share a pool.
"""

import os
import sys

os.environ.setdefault('OMP_DYNAMIC', 'FALSE')
os.environ.setdefault('MKL_DYNAMIC', 'FALSE')
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('PYTHONHASHSEED', '0')

import glob
import math
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

# ------------------------------------------------------------------------------- config
SCH_DPI, LAY_DPI = 96.0, 192.0          # verified from page sizes and from pad pitch
SCH_W, SCH_H, SCH_PXMM = 384, 288, 1.293
LAY_S, LAY_PXMM = 224, 2.0
SCH_SCALE = SCH_PXMM / (SCH_DPI / 25.4)
LAY_SCALE = LAY_PXMM / (LAY_DPI / 25.4)
SPM, LPM = SCH_DPI / 25.4, LAY_DPI / 25.4

N_MODELS = int(os.environ.get("SBL_NM",3))
EPOCHS = int(os.environ.get("SBL_EP",55))
BATCH = 48
LR = 2.0e-3
WD = 3.0e-4
DROP = 0.20
EMB = 256
AUG = 1.2
POOL_W = 1.0            # listwise term over sampled members of the query's own pool
POOL_K = 3
AUX_W = 1.0             # each tower predicts the other view's descriptors
SEED0 = 1234
THREADS = 8
RIDGE_LAM = 3.0
BLEND = 0.25            # weight of the descriptor-ridge score against the network score
SINK_TAU = 0.5          # dense Sinkhorn temperature; 0 disables
SINK_IT = 60

SCH_BG = (245, 244, 239)
SCH_PAL = [((98, 188, 96), 0), ((177, 98, 96), 1), ((255, 255, 194), 2),
           ((98, 98, 212), 3), ((0, 0, 0), 4), ((132, 0, 0), 4), ((194, 194, 0), 4)]
SCH_C = 5
LAY_PAL = [((200, 52, 52), 0), ((215, 106, 106), 0),
           ((77, 127, 196), 1), ((124, 161, 212), 1),
           ((79, 172, 227), 2), ((46, 178, 213), 2),
           ((132, 116, 220), 3),
           ((242, 237, 161), 4),
           ((232, 178, 167), 5),
           ((206, 71, 133), 6), ((2, 255, 238), 6), ((216, 100, 255), 6), ((225, 147, 202), 6),
           ((208, 210, 205), 7), ((175, 175, 175), 7), ((88, 93, 132), 7), ((255, 38, 226), 7)]
LAY_C = 8

NBIN = [1, 2, 3, 4, 6, 9, 15, 25, 45, 10 ** 9]
PAD_E = np.array([0.0, 0.05, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 1e9])
SYM_E = np.array([0.0, 20, 50, 100, 200, 400, 800, 1600, 1e9])

# ------------------------------------------------------------- palette decomposition
_Q = np.arange(32768, dtype=np.int32)
QRGB = np.stack([((_Q >> 10) & 31) * 8 + 4, ((_Q >> 5) & 31) * 8 + 4, (_Q & 31) * 8 + 4], 1).astype(np.float32)
_lut = {}


def build_lut(bg, pal, C):
    key = (bg, C)
    if key in _lut:
        return _lut[key]
    b = np.array(bg, np.float32)
    x = QRGB - b
    best_res = np.full(32768, 1e9, np.float32)
    cls = np.full(32768, C - 1, np.uint8)
    alp = np.zeros(32768, np.float32)
    for col, ch in pal:
        d = np.array(col, np.float32) - b
        n2 = float((d * d).sum())
        if n2 < 1.0:
            continue
        a = np.clip((x @ d) / n2, 0.0, 1.0)
        res = np.linalg.norm(x - a[:, None] * d[None, :], axis=1)
        m = res < best_res
        best_res[m] = res[m]
        cls[m] = ch
        alp[m] = a[m]
    ink = np.linalg.norm(x, axis=1)
    un = best_res > 60.0
    cls[un] = C - 1
    alp[un] = np.clip(ink[un] / 160.0, 0, 1)
    alp[ink < 12.0] = 0.0
    out = (cls, (alp * 255).astype(np.uint8))
    _lut[key] = out
    return out


def quant(img):
    return ((img[:, :, 0].astype(np.int32) >> 3) << 10) | \
           ((img[:, :, 1].astype(np.int32) >> 3) << 5) | (img[:, :, 2].astype(np.int32) >> 3)


def decompose(img, bg, pal, C):
    cls, alp = build_lut(bg, pal, C)
    q = quant(img)
    return cls[q], alp[q]


def pick_bg(img):
    q = quant(img)
    cnt = np.bincount(q.ravel(), minlength=32768)
    w = cnt[(31 << 10) | (31 << 5) | 31]
    if w > 0.04 * q.size and w >= cnt[0]:
        return (255, 255, 255)
    if cnt[0] > 0.04 * q.size:
        return (0, 0, 0)
    k = int(cnt.argmax())
    return (int(QRGB[k, 0]), int(QRGB[k, 1]), int(QRGB[k, 2]))


def disk(r):
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def cc_stats(mask, pxmm):
    lab, n = ndimage.label(mask)
    if n == 0:
        return [0.0] * 6
    sz = np.bincount(lab.ravel())[1:].astype(np.float64) / (pxmm * pxmm)
    sz = sz[sz > 0.02]
    if len(sz) == 0:
        return [0.0] * 6
    return [float(len(sz)), float(sz.sum()), float(np.median(sz)),
            float(np.percentile(sz, 90)), float(sz.max()), float((sz > 4.0).sum())]


K8 = np.ones((3, 3), bool)


def net_stats(mask, pxmm):
    """Component count of a connectivity mask -- wires on one view, copper on the other.
    The best-transferring cross-view descriptor found (log-log corr 0.34)."""
    lab, n = ndimage.label(mask, structure=K8)
    if n == 0:
        return [0.0] * 4
    sz = np.bincount(lab.ravel())[1:] / pxmm ** 2
    big = sz[sz > 0.5]
    return [float(n), float(len(big)), float(big.max() if len(big) else 0.0),
            float(np.median(big) if len(big) else 0.0)]


def hist_counts(v):
    h = np.zeros(len(NBIN), np.float32)
    for x in v:
        for i, e in enumerate(NBIN):
            if x <= e:
                h[i] += 1
                break
    return h


# --------------------------------------------------------------------- per-image work
def do_sch(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    H, W = img.shape[:2]
    cls, alp = decompose(img, SCH_BG, SCH_PAL, SCH_C)
    s = min(SCH_SCALE, SCH_W / W, SCH_H / H)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    canvas = np.zeros((SCH_H, SCH_W, SCH_C), np.uint8)
    y0, x0 = (SCH_H - nh) // 2, (SCH_W - nw) // 2
    meta = [W / SPM, H / SPM, s / SCH_SCALE]
    for ch in range(SCH_C):
        m = np.where(cls == ch, alp, 0)
        meta.append(float(m.sum()) / 255.0 / SPM ** 2)
        canvas[y0:y0 + nh, x0:x0 + nw, ch] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    body = (cls == 2) & (alp > 110)
    ink = (cls == 1) & (alp > 110)
    wire = (cls == 0) & (alp > 60)
    meta += cc_stats(body, SPM) + cc_stats(ink, SPM)

    st = [W / SPM, H / SPM, W * H / SPM ** 2, float(wire.sum()) / SPM / 0.6]
    bl, bn = ndimage.label(body)
    bar = np.bincount(bl.ravel())[1:] / SPM ** 2 if bn else np.zeros(0)
    good = np.nonzero(bar > 8.0)[0] + 1
    pins = np.zeros(len(good), np.float32)
    if len(good):
        objs = ndimage.find_objects(bl)
        near_k, far_k = disk(0.7 * SPM), disk(2.6 * SPM)
        pad = int(far_k.shape[0] // 2 + 2)
        for t, g in enumerate(good):
            sl = objs[g - 1]
            y1, y2 = max(0, sl[0].start - pad), min(H, sl[0].stop + pad)
            x1, x2 = max(0, sl[1].start - pad), min(W, sl[1].stop + pad)
            m = bl[y1:y2, x1:x2] == g
            ring = ndimage.binary_dilation(m, far_k) & ~ndimage.binary_dilation(m, near_k)
            _, ns = ndimage.label(ring & ink[y1:y2, x1:x2])
            pins[t] = ns
    small = ink & ~ndimage.binary_dilation(body, disk(0.5 * SPM))
    slab, sn = ndimage.label(small)
    sar = np.bincount(slab.ravel())[1:] / SPM ** 2 if sn else np.zeros(0)
    sar = sar[(sar > 0.3) & (sar < 60.0)]
    st += [float(len(good)), float(bar[bar > 8].sum() if bn else 0), float(pins.sum()),
           float(pins.max() if len(pins) else 0), float((pins >= 8).sum()),
           float(len(sar)), float(sar.sum()), float(np.median(sar) if len(sar) else 0)]
    st += list(hist_counts(pins)) + list(np.histogram(bar[bar > 8] if bn else np.zeros(0), bins=SYM_E)[0])
    # every wired pin shows up as one wire/symbol contact blob -- the most reliable global
    # pin-count proxy available; per-body stub attribution undercounts badly
    contact = ndimage.binary_dilation(wire, K8) & ndimage.binary_dilation(ink, K8)
    _, ncon = ndimage.label(contact, structure=K8)
    st += [float(ncon)]
    st += net_stats(ndimage.binary_closing(wire, K8), SPM)
    st += net_stats(ndimage.binary_dilation(wire | ink, K8), SPM)
    return canvas, np.array(meta + st, np.float32)


def do_lay(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    H, W = img.shape[:2]
    cls, alp = decompose(img, pick_bg(img), LAY_PAL, LAY_C)
    s = min(LAY_SCALE, LAY_S / W, LAY_S / H)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    canvas = np.zeros((LAY_S, LAY_S, LAY_C), np.uint8)
    y0, x0 = (LAY_S - nh) // 2, (LAY_S - nw) // 2
    meta = [W / LPM, H / LPM, s / LAY_SCALE]
    for ch in range(LAY_C):
        m = np.where(cls == ch, alp, 0)
        meta.append(float(m.sum()) / 255.0 / LPM ** 2)
        canvas[y0:y0 + nh, x0:x0 + nw, ch] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    meta += cc_stats((cls == 2) & (alp > 110), LPM)
    meta += cc_stats((cls == 3) & (alp > 110), LPM)
    meta += cc_stats((cls == 4) & (alp > 110), LPM)

    pad = ((cls == 2) | (cls == 3)) & (alp > 110)
    cu = ((cls == 0) | (cls == 1) | (cls == 2) | (cls == 3)) & (alp > 90)
    fr = ((cls == 0) | (cls == 2) | (cls == 3)) & (alp > 90)
    nets = net_stats(cu, LPM) + net_stats(fr, LPM)
    st = [W / LPM, H / LPM, W * H / LPM ** 2]
    if pad.sum() < 3:
        st += [0.0] * (8 + len(NBIN) + len(PAD_E) - 1)
        return canvas, np.array(meta + st + nets, np.float32)
    lab, n = ndimage.label(pad)
    ar = np.bincount(lab.ravel())[1:] / LPM ** 2
    keep = ar > 0.03
    cen = np.array(ndimage.center_of_mass(pad, lab, np.nonzero(keep)[0] + 1)) if keep.any() else np.zeros((0, 2))
    ar = ar[keep]
    gl, gn = ndimage.label(ndimage.binary_dilation(pad, disk(0.45 * LPM)))
    if len(cen):
        gid = gl[np.clip(cen[:, 0].astype(int), 0, H - 1), np.clip(cen[:, 1].astype(int), 0, W - 1)]
        per = np.bincount(gid[gid > 0], minlength=gn + 1)[1:]
        per = per[per > 0]
    else:
        per = np.zeros(0)
    st += [float(len(ar)), float(ar.sum()), float(len(per)), float(np.median(ar) if len(ar) else 0),
           float(np.percentile(ar, 90) if len(ar) else 0), float(ar.max() if len(ar) else 0),
           float((per >= 8).sum()), float(per.max() if len(per) else 0)]
    st += list(hist_counts(per)) + list(np.histogram(ar, bins=PAD_E)[0])
    return canvas, np.array(meta + st + nets, np.float32)


def prepare(paths, fn, C, hh, ww, tag):
    # memmap-backed: the two rasters are ~5.4 GB together and the training loop only ever
    # touches a batch at a time, so they do not need to be resident
    n = len(paths)
    os.makedirs('working', exist_ok=True)
    arr = np.lib.format.open_memmap(os.path.join('working', tag + '.npy'), 'w+', np.uint8, (n, hh, ww, C))
    metas = [None] * n

    def job(i):
        c, m = fn(paths[i])
        arr[i] = c
        metas[i] = m
    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(job, range(n)))
    arr.flush()
    return arr, np.stack(metas)


# ------------------------------------------------------------------------------ model
class Tower(nn.Module):
    def __init__(s, cin, nmeta, naux, widths=(32, 64, 96, 128, 192, 256), emb=EMB, drop=DROP):
        super().__init__()
        L, c = [], cin
        for i, w in enumerate(widths):
            L += [nn.Conv2d(c, w, 3, 2 if i < 5 else 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True),
                  nn.Conv2d(w, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(True)]
            c = w
        s.body = nn.Sequential(*L)
        s.drop = nn.Dropout(drop)
        s.mlp = nn.Sequential(nn.Linear(nmeta, 128), nn.BatchNorm1d(128), nn.ReLU(True),
                              nn.Linear(128, 128), nn.ReLU(True))
        s.trunk = nn.Sequential(nn.Linear(2 * c + 128, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        s.emb = nn.Linear(512, emb)
        s.aux = nn.Linear(512, naux)

    def forward(s, x, m, aux=False):
        h = s.body(x)
        g = torch.cat([h.mean((2, 3)), h.amax((2, 3)), s.mlp(m)], 1)
        t = s.trunk(s.drop(g))
        e = F.normalize(s.emb(t), dim=1)
        return (e, s.aux(t)) if aux else e


class Duo(nn.Module):
    def __init__(s, cs, ms, cl, ml):
        super().__init__()
        s.a = Tower(cs, ms, ml)
        s.b = Tower(cl, ml, ms)
        s.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def scale(s):
        return s.logit_scale.exp().clamp(max=100.0)


# ------------------------------------------------------------------------- augmentation
def jitter(x, s):
    B, dev = x.shape[0], x.device
    sc = 1.0 + (torch.rand(B, device=dev) - 0.5) * 0.16 * s
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = sc
    th[:, 1, 1] = sc
    th[:, :, 2] = (torch.rand(B, 2, device=dev) - 0.5) * 0.06 * s
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False),
                         align_corners=False, padding_mode='zeros')


def aug_l(x, s):
    B, dev = x.shape[0], x.device
    f = torch.rand(B, device=dev) < 0.5
    x = torch.where(f[:, None, None, None], x.flip(3), x)
    f = torch.rand(B, device=dev) < 0.5
    x = torch.where(f[:, None, None, None], x.flip(2), x)
    f = torch.rand(B, device=dev) < 0.5
    x = torch.where(f[:, None, None, None], x.transpose(2, 3), x)
    return jitter(x, s)


# ---------------------------------------------------------------------------- training
def train_one(S, L, SM, LM, qidx, tidx, cand, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    n = len(qidx)
    model = Duo(S.shape[3], SM.shape[1], L.shape[3], LM.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * math.ceil(n / BATCH)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(device, enabled=(device == 'cuda'))
    SMg = torch.from_numpy(SM).to(device)
    LMg = torch.from_numpy(LM).to(device)
    rng = np.random.RandomState(seed)

    def get(arr, ix):
        return torch.from_numpy(np.ascontiguousarray(arr[ix])).to(device).permute(0, 3, 1, 2).float().div_(255)

    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 8:
                continue
            qi, ti = qidx[b], tidx[b]
            xs = jitter(get(S, qi), AUG * 0.6)
            xl = aug_l(get(L, ti), AUG)
            gs = SMg[torch.from_numpy(qi).to(device)]
            gl = LMg[torch.from_numpy(ti).to(device)]
            with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == 'cuda')):
                ea, pa = model.a(xs, gs, aux=True)
                eb, pb = model.b(xl, gl, aux=True)
                lg = model.scale() * ea @ eb.t()
                tgt = torch.arange(len(b), device=device)
                loss = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.t(), tgt))
                loss = loss + AUX_W * 0.5 * (F.smooth_l1_loss(pa, gl) + F.smooth_l1_loss(pb, gs))
                cb = cand[b]
                msk = cb != ti[:, None]
                pick = np.stack([rng.choice(cb[r][msk[r]], POOL_K, replace=msk[r].sum() < POOL_K)
                                 for r in range(len(b))])
                flat = pick.reshape(-1)
                ec = model.b(aug_l(get(L, flat), AUG),
                             LMg[torch.from_numpy(flat).to(device)]).view(len(b), POOL_K, -1)
                neg = model.scale() * (ea[:, None, :] * ec).sum(-1)
                pos = model.scale() * (ea * eb).sum(-1, keepdim=True)
                loss = loss + POOL_W * F.cross_entropy(
                    torch.cat([pos, neg], 1), torch.zeros(len(b), dtype=torch.long, device=device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
        print('    epoch %d/%d' % (ep + 1, EPOCHS), flush=True)
    return model


@torch.no_grad()
def embed(model, which, arr, M, device, tta=False, bs=64):
    tower = getattr(model, which)
    model.eval()
    Mg = torch.from_numpy(M).to(device)
    out = []
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[i:i + bs])).to(device).permute(0, 3, 1, 2).float().div_(255)
        m = Mg[i:i + bs]
        with torch.amp.autocast(device, dtype=torch.float16, enabled=(device == 'cuda')):
            e = tower(x, m).float()
            if tta:
                for t in (x.flip(3), x.flip(2), x.flip(2).flip(3)):
                    e = e + tower(t, m).float()
        out.append(F.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()


def dense_sinkhorn(S, iters, tau):
    L = S.astype(np.float64) / tau
    for _ in range(iters):
        mx = L.max(1, keepdims=True)
        L = L - (mx + np.log(np.exp(L - mx).sum(1, keepdims=True)))
        mx = L.max(0, keepdims=True)
        L = L - (mx + np.log(np.exp(L - mx).sum(0, keepdims=True)))
    return L


# -------------------------------------------------------------------------------- main
def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(THREADS)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cc = ['cand%02d' % i for i in range(1, 21)]
    rr = ['rel%02d' % i for i in range(1, 21)]
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    # a valid submission exists from here on, whatever happens later
    os.makedirs('working', exist_ok=True)
    flat = pd.DataFrame(np.full((len(test), 20), 0.5), columns=rr)
    flat.insert(0, 'id', test['id'].values)
    flat.to_csv(out_path, index=False)

    sch_ids = list(train['id']) + list(test['id'])
    lay_tr = sorted({v for v in train[cc].values.ravel()})
    lay_te = sorted({v for v in test[cc].values.ravel()})
    print('train %d test %d | layouts %d/%d | device %s'
          % (len(train), len(test), len(lay_tr), len(lay_te), device), flush=True)

    S, SM = prepare([os.path.join(data_dir, 'schematics', i + '.png') for i in sch_ids],
                    do_sch, SCH_C, SCH_H, SCH_W, 'sch')
    L, LM = prepare([os.path.join(data_dir, 'train', 'layouts', i + '.png') for i in lay_tr] +
                    [os.path.join(data_dir, 'test', 'layouts', i + '.png') for i in lay_te],
                    do_lay, LAY_C, LAY_S, LAY_S, 'lay')
    print('rasterised %s %s' % (S.shape, L.shape), flush=True)

    SMn = np.log1p(np.maximum(SM, 0.0))
    LMn = np.log1p(np.maximum(LM, 0.0))
    SMn = ((SMn - SMn.mean(0)) / (SMn.std(0) + 1e-6)).astype(np.float32)
    LMn = ((LMn - LMn.mean(0)) / (LMn.std(0) + 1e-6)).astype(np.float32)

    SI = {v: i for i, v in enumerate(sch_ids)}
    LI = {v: i for i, v in enumerate(lay_tr + lay_te)}
    qidx = np.array([SI[i] for i in train['id']])
    cand = np.vectorize(LI.get)(train[cc].values)
    tidx = cand[np.arange(len(cand)), train[rr].values.argmax(1)]
    qtest = np.array([SI[i] for i in test['id']])
    ctest = np.vectorize(LI.get)(test[cc].values)
    lte = np.array([LI[i] for i in lay_te])

    acc = np.zeros((len(test), len(lte)))
    for k in range(N_MODELS):
        model = train_one(S, L, SMn, LMn, qidx, tidx, cand, SEED0 + 101 * k, device)
        EQ = embed(model, 'a', S[qtest], SMn[qtest], device)
        EL = embed(model, 'b', L[lte], LMn[lte], device, tta=True)
        acc += EQ @ EL.T
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        print('  model %d/%d done' % (k + 1, N_MODELS), flush=True)
    net = acc / N_MODELS

    # descriptor ridge: schematic descriptors -> layout descriptors, cosine scored
    A = np.c_[SMn[qidx], np.ones(len(qidx))]
    W = np.linalg.solve(A.T @ A + RIDGE_LAM * np.eye(A.shape[1]), A.T @ LMn[tidx])
    P = np.c_[SMn[qtest], np.ones(len(qtest))] @ W
    P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    Ln = LMn[lte] / (np.linalg.norm(LMn[lte], axis=1, keepdims=True) + 1e-9)
    ridge = P @ Ln.T

    dense = (1 - BLEND) * net + BLEND * ridge
    if SINK_TAU > 0:
        dense = dense_sinkhorn(dense, SINK_IT, SINK_TAU)

    pos = {v: i for i, v in enumerate(lte)}
    col = np.vectorize(pos.get)(ctest)
    Sc = dense[np.arange(len(test))[:, None], col]

    Sc = Sc - Sc.min(1, keepdims=True)
    Sc = Sc / np.maximum(Sc.max(1, keepdims=True), 1e-9)
    Sc = np.clip(np.nan_to_num(Sc, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)

    sub = pd.DataFrame(Sc, columns=rr)
    sub.insert(0, 'id', test['id'].values)
    sub = sub[['id'] + rr]
    sub.to_csv(out_path, index=False)
    sub.to_csv(os.path.join('working', 'submission.csv'), index=False)
    print('wrote %s %s' % (out_path, sub.shape), flush=True)


if __name__ == '__main__':
    main()
