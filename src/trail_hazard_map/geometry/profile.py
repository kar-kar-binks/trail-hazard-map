import numpy as np


def interp_fill(a):
    a = a.astype(float).copy()
    bad = ~np.isfinite(a)
    if bad.all():
        return np.zeros_like(a)
    a[bad] = np.interp(np.nonzero(bad)[0], np.nonzero(~bad)[0], a[~bad])
    return a


def binned(values, bins, nb, fn):
    out = np.full(nb, np.nan)
    order = np.argsort(bins, kind="stable")
    sb = bins[order]
    cuts = np.searchsorted(sb, np.arange(nb + 1))
    for i in range(nb):
        seg = values[order[cuts[i] : cuts[i + 1]]]
        if len(seg):
            out[i] = fn(seg)
    return out


def ground_profile(s, d, z, trail_like, length, fine=0.5):
    nb = int(np.ceil(length / fine))
    sel = trail_like & (np.abs(d) < 0.7)
    b = np.clip((s[sel] / fine).astype(int), 0, nb - 1)
    gz = interp_fill(binned(z[sel], b, nb, np.median))
    centers = (np.arange(nb) + 0.5) * fine
    return centers, gz


def segment_profile(s, d, z, opacity, trail_like, gz_s, gz, length, seg_m, cfg):
    g = cfg["geometry"]
    nb = int(np.ceil(length / seg_m))
    b = np.clip((s / seg_m).astype(int), 0, nb - 1)
    hh = np.interp(s, gz_s, gz)
    tl = trail_like & (np.abs(d) < 4.0)
    width = interp_fill(binned(d[tl], b[tl], nb, lambda v: np.percentile(v, 97.5) - np.percentile(v, 2.5)))
    half = width[b] / 2
    drops = []
    for sign in (1, -1):
        band = (sign * d > half + 1.0) & (sign * d < half + 2.5)
        low = interp_fill(binned(z[band] - hh[band], b[band], nb, lambda v: np.percentile(v, 10)))
        drops.append(np.maximum(0, -low))
    dropoff = np.maximum(drops[0], drops[1])
    zf = np.interp((np.arange(nb) + 0.5) * seg_m, gz_s, gz)
    gradient = np.gradient(gz, gz_s)
    per = int(round(seg_m / 0.5))
    step = np.zeros(nb)
    for i in range(nb):
        lo, hi = i * per, min((i + 1) * per + 1, len(gz))
        jump = np.abs(np.diff(gz[lo:hi]) - np.median(np.diff(gz)))
        step[i] = jump.max() if len(jump) else 0.0
    cl = (np.abs(d) < half) & (opacity > 0.5) & ((z - hh) > 0.7)
    clearance = np.full(nb, g["clearance_cap_m"])
    low = binned(z[cl] - hh[cl], b[cl], nb, np.min)
    clearance = np.where(np.isfinite(low), np.minimum(low, g["clearance_cap_m"]), clearance)
    return {"width": width, "dropoff": dropoff, "step": step, "clearance": clearance, "ground": zf, "slope_profile": gradient}


def sight_distance(xyz, opacity, line, line_s, gz_s, gz, length, seg_m, cfg, res=0.5):
    g = cfg["geometry"]
    keep = opacity > 0.5
    pts = xyz[keep]
    lo = pts[:, :2].min(0) - 1
    shape = np.ceil((pts[:, :2].max(0) + 1 - lo) / res).astype(int)
    hf = np.full(shape[::-1], -np.inf)
    ij = ((pts[:, :2] - lo) / res).astype(int)
    np.maximum.at(hf, (ij[:, 1], ij[:, 0]), pts[:, 2])
    nb = int(np.ceil(length / seg_m))
    out = np.zeros(nb)
    ahead = np.arange(2.0, g["max_sight_m"] + 1, 1.0)
    for i in range(nb):
        s0 = (i + 0.5) * seg_m
        p0 = np.array([np.interp(s0, line_s, line[:, 0]), np.interp(s0, line_s, line[:, 1])])
        ez = np.interp(s0, gz_s, gz) + g["eye_height_m"]
        reach = g["max_sight_m"]
        for a in ahead:
            s1 = min(s0 + a, line_s[-1])
            p1 = np.array([np.interp(s1, line_s, line[:, 0]), np.interp(s1, line_s, line[:, 1])])
            tz = np.interp(s1, gz_s, gz) + 0.3
            t = np.linspace(0.08, 0.98, 14)
            xy = p0 + t[:, None] * (p1 - p0)
            rz = ez + t * (tz - ez)
            cell = ((xy - lo) / res).astype(int)
            ok = (cell[:, 0] >= 0) & (cell[:, 0] < shape[0]) & (cell[:, 1] >= 0) & (cell[:, 1] < shape[1])
            h = np.where(ok, hf[np.clip(cell[:, 1], 0, shape[1] - 1), np.clip(cell[:, 0], 0, shape[0] - 1)], -np.inf)
            if np.any(h > rz + 0.2):
                reach = a
                break
        out[i] = reach
    return out
