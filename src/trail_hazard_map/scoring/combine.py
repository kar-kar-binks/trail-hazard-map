import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .terms import robust_norm

UNIT_TERMS = ("hazard", "traction", "wet")


def combine(gterms, sterms, seg_of, corridor, cfg):
    sc = cfg["scoring"]
    w = sc["weights"]
    lo, hi = sc["percentile_lo"], sc["percentile_hi"]
    normed = {}
    for k, v in gterms.items():
        if k in UNIT_TERMS:
            normed[k] = v
        else:
            ref = v[corridor]
            lo_v, hi_v = np.percentile(ref, [lo, hi])
            normed[k] = np.clip((v - lo_v) / max(hi_v - lo_v, sc["floors"][k]), 0, 1)
    for k, v in sterms.items():
        sm = ndimage.gaussian_filter1d(robust_norm(v, sc["floors"][k], lo, hi), sc["smooth_sigma_segments"], mode="nearest")
        normed[k] = sm[seg_of]
    contrib = {k: w[k] * normed[k] for k in w}
    return normed, contrib


def total(contrib):
    return sum(contrib.values())


def smooth3d(risk, xyz, k):
    tree = cKDTree(xyz)
    _, nbr = tree.query(xyz, k=k)
    return risk[nbr].mean(1)


def segment_scores(risk, seg_of, corridor, nb, pct=90):
    out = np.zeros(nb)
    order = np.argsort(seg_of[corridor], kind="stable")
    idx = np.nonzero(corridor)[0][order]
    cuts = np.searchsorted(seg_of[idx], np.arange(nb + 1))
    for i in range(nb):
        sel = idx[cuts[i] : cuts[i + 1]]
        out[i] = np.percentile(risk[sel], pct) if len(sel) else 0.0
    return out


def worst_window(seg_risk, seg_m, window_m):
    n = max(1, int(round(window_m / seg_m)))
    kern = np.ones(n) / n
    avg = np.convolve(seg_risk, kern, mode="valid")
    i = int(avg.argmax())
    return i * seg_m, (i + n) * seg_m, float(avg[i])
