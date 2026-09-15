"""Palette-aware layer decomposition + fixed-physical-scale rasterisation."""
import os, sys, glob, json, time
import numpy as np, cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage

DATA = sys.argv[1] if len(sys.argv) > 1 else 'C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_DATA'
OUT  = sys.argv[2] if len(sys.argv) > 2 else 'C:/Users/tarun/OneDrive/Desktop/ERIS/SBL_WORK/cache'
os.makedirs(OUT, exist_ok=True)

SCH_DPI, LAY_DPI = 96.0, 192.0          # verified: A4 sheets = 1123x794 px; 2.54 mm pad pitch = 19.2 px
SCH_W = int(os.environ.get('SBL_SW', 384)); SCH_H = int(os.environ.get('SBL_SH', 288))
SCH_PXMM = float(os.environ.get('SBL_SPX', 1.293))
LAY_S = int(os.environ.get('SBL_LS', 224)); LAY_PXMM = float(os.environ.get('SBL_LPX', 2.0))
SCH_SCALE = SCH_PXMM / (SCH_DPI / 25.4)
LAY_SCALE = LAY_PXMM / (LAY_DPI / 25.4)

SCH_BG = (245, 244, 239)
# (colour, channel)
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

_Q = np.arange(32768, dtype=np.int32)
QR = ((_Q >> 10) & 31).astype(np.float32) * 8 + 4
QG = ((_Q >> 5) & 31).astype(np.float32) * 8 + 4
QB = (_Q & 31).astype(np.float32) * 8 + 4
QRGB = np.stack([QR, QG, QB], 1)

_lut_cache = {}


def build_lut(bg, pal, C):
    key = (bg, id(pal))
    if key in _lut_cache:
        return _lut_cache[key]
    b = np.array(bg, np.float32)
    x = QRGB - b                                        # 32768 x 3
    best_res = np.full(32768, 1e9, np.float32)
    best_cls = np.full(32768, 255, np.uint8)
    best_a = np.zeros(32768, np.float32)
    for col, ch in pal:
        d = np.array(col, np.float32) - b
        n2 = float((d * d).sum())
        if n2 < 1.0:
            continue
        a = np.clip((x @ d) / n2, 0.0, 1.0)
        res = np.linalg.norm(x - a[:, None] * d[None, :], axis=1)
        m = res < best_res
        best_res[m] = res[m]
        best_cls[m] = ch
        best_a[m] = a[m]
    ink = np.linalg.norm(x, axis=1)                     # distance from background
    unexplained = best_res > 60.0                       # colour no palette entry explains
    cls = best_cls.copy()
    alp = best_a.copy()
    cls[unexplained] = C - 1                            # catch-all channel
    alp[unexplained] = np.clip(ink[unexplained] / 160.0, 0, 1)
    faint = ink < 12.0
    alp[faint] = 0.0
    out = (cls, (alp * 255).astype(np.uint8))
    _lut_cache[key] = out
    return out


def quant(img):
    return ((img[:, :, 0].astype(np.int32) >> 3) << 10) | \
           ((img[:, :, 1].astype(np.int32) >> 3) << 5) | (img[:, :, 2].astype(np.int32) >> 3)


def decompose(img, bg, pal, C):
    cls, alp = build_lut(bg, pal, C)
    q = quant(img)
    c = cls[q]
    a = alp[q]
    return c, a


def pick_bg(img):
    q = quant(img)
    cnt = np.bincount(q.ravel(), minlength=32768)
    w = cnt[(31 << 10) | (31 << 5) | 31]
    k = cnt.argmax()
    if w > 0.04 * q.size and w >= cnt[0]:
        return (255, 255, 255)
    if cnt[0] > 0.04 * q.size:
        return (0, 0, 0)
    return (int(QR[k]), int(QG[k]), int(QB[k]))


def cc_stats(mask, pxmm):
    lab, n = ndimage.label(mask)
    if n == 0:
        return [0.0] * 6
    sz = np.bincount(lab.ravel())[1:].astype(np.float64) / (pxmm * pxmm)   # mm^2
    sz = sz[sz > 0.02]
    if len(sz) == 0:
        return [0.0] * 6
    return [float(len(sz)), float(sz.sum()), float(np.median(sz)),
            float(np.percentile(sz, 90)), float(sz.max()), float((sz > 4.0).sum())]


def process(args):
    path, kind = args
    img = np.asarray(Image.open(path).convert('RGB'))
    H, W = img.shape[:2]
    if kind == 's':
        bg, pal, C, scale = SCH_BG, SCH_PAL, SCH_C, SCH_SCALE
        cw, chh, dpi, pxmm = SCH_W, SCH_H, SCH_DPI, SCH_PXMM
    else:
        bg, pal, C, scale = pick_bg(img), LAY_PAL, LAY_C, LAY_SCALE
        cw, chh, dpi, pxmm = LAY_S, LAY_S, LAY_DPI, LAY_PXMM
    c, a = decompose(img, bg, pal, C)
    s = min(scale, cw / W, chh / H)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    canvas = np.zeros((chh, cw, C), np.uint8)
    y0, x0 = (chh - nh) // 2, (cw - nw) // 2
    meta = [W / dpi * 25.4, H / dpi * 25.4, s / scale]
    for ch in range(C):
        m = np.where(c == ch, a, 0)
        meta.append(float(m.sum()) / 255.0 / (dpi / 25.4) ** 2)          # mm^2 of ink
        canvas[y0:y0 + nh, x0:x0 + nw, ch] = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    if kind == 's':
        meta += cc_stats((c == 2) & (a > 128), dpi / 25.4)               # symbol bodies
        meta += cc_stats((c == 1) & (a > 128), dpi / 25.4)               # outlines/pins
    else:
        meta += cc_stats((c == 2) & (a > 128), dpi / 25.4)               # through-hole pads/vias
        meta += cc_stats((c == 3) & (a > 128), dpi / 25.4)               # smd pads
        meta += cc_stats((c == 4) & (a > 128), dpi / 25.4)               # front silkscreen
    return canvas, np.array(meta, np.float32)


def run(paths, kind, tag, C, hh, ww):
    n = len(paths)
    arr = np.lib.format.open_memmap(os.path.join(OUT, tag + '.npy'), 'w+', np.uint8, (n, hh, ww, C))
    metas = [None] * n
    t0 = time.time()
    def job(i):
        cv, mt = process((paths[i], kind))
        arr[i] = cv
        metas[i] = mt
    with ThreadPoolExecutor(20) as ex:
        for k, _ in enumerate(ex.map(job, range(n))):
            if k % 1000 == 0:
                print('  %s %d/%d  %.0fs' % (tag, k, n, time.time() - t0), flush=True)
    arr.flush()
    np.save(os.path.join(OUT, tag + '_meta.npy'), np.stack(metas))
    print('%s done %d in %.0fs' % (tag, n, time.time() - t0), flush=True)


if __name__ == '__main__':
    sch = sorted(glob.glob(os.path.join(DATA, 'schematics', '*.png')))
    lay = sorted(glob.glob(os.path.join(DATA, 'train', 'layouts', '*.png'))) + \
          sorted(glob.glob(os.path.join(DATA, 'test', 'layouts', '*.png')))
    json.dump({'sch': [os.path.splitext(os.path.basename(p))[0] for p in sch],
               'lay': [os.path.splitext(os.path.basename(p))[0] for p in lay]},
              open(os.path.join(OUT, 'index.json'), 'w'))
    run(sch, 's', 'sch', SCH_C, SCH_H, SCH_W)
    run(lay, 'l', 'lay', LAY_C, LAY_S, LAY_S)
