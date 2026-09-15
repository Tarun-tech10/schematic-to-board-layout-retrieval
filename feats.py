"""Native-resolution structural descriptors: footprint pad-counts vs symbol pin-counts."""
import os, sys, glob, json, time
import numpy as np
from PIL import Image
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor
import prep as P

DATA = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'
OUT  = sys.argv[2] if len(sys.argv) > 2 else 'cache'
SPM, LPM = P.SCH_DPI/25.4, P.LAY_DPI/25.4          # px per mm in each view
NBIN = [1, 2, 3, 4, 6, 9, 15, 25, 45, 10**9]        # pins/pads per part, upper edges

def hist_counts(v):
    h = np.zeros(len(NBIN), np.float32)
    for x in v:
        for i, e in enumerate(NBIN):
            if x <= e:
                h[i] += 1; break
    return h

def areahist(a, edges):
    return np.histogram(a, bins=edges)[0].astype(np.float32)

PAD_E = np.array([0.0, 0.05, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 1e9])   # mm^2
SYM_E = np.array([0.0, 20, 50, 100, 200, 400, 800, 1600, 1e9])             # mm^2

def disk(r):
    r = max(1, int(round(r)))
    y, x = np.ogrid[-r:r+1, -r:r+1]
    return (x*x + y*y) <= r*r

def lay_feat(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    H, W = img.shape[:2]
    cls, alp = P.decompose(img, P.pick_bg(img), P.LAY_PAL, P.LAY_C)
    pad = ((cls == 2) | (cls == 3)) & (alp > 110)
    out = [W/LPM, H/LPM, W*H/LPM**2]
    if pad.sum() < 3:
        return np.array(out + [0.0]*(len(NBIN)+len(PAD_E)-1+8), np.float32)
    lab, n = ndimage.label(pad)
    ar = np.bincount(lab.ravel())[1:] / LPM**2
    keep = ar > 0.03
    cen = np.array(ndimage.center_of_mass(pad, lab, np.nonzero(keep)[0]+1)) if keep.any() else np.zeros((0, 2))
    ar = ar[keep]
    grp = ndimage.binary_dilation(pad, disk(0.45*LPM))
    gl, gn = ndimage.label(grp)
    if len(cen):
        gid = gl[np.clip(cen[:, 0].astype(int), 0, H-1), np.clip(cen[:, 1].astype(int), 0, W-1)]
        per = np.bincount(gid[gid > 0], minlength=gn+1)[1:]
        per = per[per > 0]
    else:
        per = np.zeros(0)
    out += [float(len(ar)), float(ar.sum()), float(len(per)), float(np.median(ar) if len(ar) else 0),
            float(np.percentile(ar, 90) if len(ar) else 0), float(ar.max() if len(ar) else 0),
            float((per >= 8).sum()), float(per.max() if len(per) else 0)]
    return np.array(out + list(hist_counts(per)) + list(areahist(ar, PAD_E)), np.float32)

def sch_feat(path):
    img = np.asarray(Image.open(path).convert('RGB'))
    H, W = img.shape[:2]
    cls, alp = P.decompose(img, P.SCH_BG, P.SCH_PAL, P.SCH_C)
    body = (cls == 2) & (alp > 110)
    ink  = (cls == 1) & (alp > 110)
    wire = (cls == 0) & (alp > 60)
    out = [W/SPM, H/SPM, W*H/SPM**2, float(wire.sum())/SPM/0.6]     # wire length proxy, mm
    bl, bn = ndimage.label(body)
    bar = np.bincount(bl.ravel())[1:]/SPM**2 if bn else np.zeros(0)
    good = np.nonzero(bar > 8.0)[0] + 1
    pins = np.zeros(len(good), np.float32)
    if len(good):
        keep = np.isin(bl, good)
        near = ndimage.binary_dilation(keep, disk(0.7*SPM))
        far  = ndimage.binary_dilation(keep, disk(2.6*SPM))
        stub = (far & ~near) & ink
        sl, sn = ndimage.label(stub)
        if sn:
            _, ii = ndimage.distance_transform_edt(~keep, return_indices=True)
            own = bl[ii[0], ii[1]]
            sc = ndimage.center_of_mass(stub, sl, range(1, sn+1))
            sc = np.array(sc).astype(int)
            oid = own[np.clip(sc[:, 0], 0, H-1), np.clip(sc[:, 1], 0, W-1)]
            cnt = np.bincount(oid, minlength=bn+1)
            pins = cnt[good].astype(np.float32)
    small = ink & ~ndimage.binary_dilation(body, disk(0.5*SPM))
    slab, sn2 = ndimage.label(small)
    sar = np.bincount(slab.ravel())[1:]/SPM**2 if sn2 else np.zeros(0)
    sar = sar[(sar > 0.3) & (sar < 60.0)]
    out += [float(len(good)), float(bar[bar > 8].sum() if bn else 0), float(pins.sum()),
            float(pins.max() if len(pins) else 0), float((pins >= 8).sum()),
            float(len(sar)), float(sar.sum()), float(np.median(sar) if len(sar) else 0)]
    return np.array(out + list(hist_counts(pins)) + list(areahist(bar[bar > 8] if bn else np.zeros(0), SYM_E)), np.float32)

def run(paths, fn, tag):
    res = [None]*len(paths); t0 = time.time()
    def job(i):
        try: res[i] = fn(paths[i])
        except Exception as e: res[i] = None; print('ERR', paths[i], e, flush=True)
    with ThreadPoolExecutor(20) as ex:
        for k, _ in enumerate(ex.map(job, range(len(paths)))):
            if k % 1500 == 0: print('  %s %d/%d %.0fs' % (tag, k, len(paths), time.time()-t0), flush=True)
    d = max(len(x) for x in res if x is not None)
    A = np.stack([x if x is not None else np.zeros(d, np.float32) for x in res])
    np.save(os.path.join(OUT, tag+'.npy'), A)
    print('%s %s %.0fs' % (tag, A.shape, time.time()-t0), flush=True)

if __name__ == '__main__':
    sch = sorted(glob.glob(os.path.join(DATA, 'schematics', '*.png')))
    lay = sorted(glob.glob(os.path.join(DATA, 'train', 'layouts', '*.png'))) + \
          sorted(glob.glob(os.path.join(DATA, 'test', 'layouts', '*.png')))
    run(lay, lay_feat, 'lay_struct')
    run(sch, sch_feat, 'sch_struct')
