import numpy as np
from scipy import ndimage

from .filters import convolve, gaussian_blur


def harris(gray, k=0.05, sigma=1.5, n_best=900, border=8, nms=5):
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / 8.0
    ix = convolve(gray, sx)
    iy = convolve(gray, sx.T)
    sxx = gaussian_blur(ix * ix, sigma)
    syy = gaussian_blur(iy * iy, sigma)
    sxy = gaussian_blur(ix * iy, sigma)
    resp = sxx * syy - sxy ** 2 - k * (sxx + syy) ** 2
    peaks = (resp == ndimage.maximum_filter(resp, size=nms)) & (resp > 1e-4 * resp.max())
    peaks[:border] = peaks[-border:] = False
    peaks[:, :border] = peaks[:, -border:] = False
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-resp[ys, xs])[:n_best]
    return np.stack([xs[order], ys[order]], 1).astype(float)


def describe(gray, pts, r=6):
    descs, kept = [], []
    for x, y in pts.astype(int):
        p = gray[y - r : y + r + 1, x - r : x + r + 1]
        if p.shape != (2 * r + 1, 2 * r + 1):
            continue
        v = p.ravel() - p.mean()
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        descs.append(v / n)
        kept.append((x, y))
    return np.array(kept, float), np.array(descs)


def match_ratio(d1, d2, ratio=0.85):
    sim = d1 @ d2.T
    order = np.argsort(-sim, axis=1)
    best = order[:, 0]
    second = order[:, 1]
    rows = np.arange(len(d1))
    dist_best = np.sqrt(np.maximum(2 - 2 * sim[rows, best], 0))
    dist_second = np.sqrt(np.maximum(2 - 2 * sim[rows, second], 0))
    ok = dist_best < ratio * dist_second
    back = np.argmax(sim, axis=0)
    ok &= back[best] == rows
    return np.stack([rows[ok], best[ok]], 1)
