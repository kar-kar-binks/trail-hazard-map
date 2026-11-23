import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def thin(mask):
    img = mask.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            n = [P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]]
            B = sum(n)
            A = sum(((n[i] == 0) & (n[(i + 1) % 8] == 1)).astype(int) for i in range(8))
            if step == 0:
                c = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                c = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
    return img.astype(bool)


def extract(xyz, trail_like, res=0.5, step=0.5):
    pts = xyz[trail_like, :2]
    lo = pts.min(0) - 2
    hi = pts.max(0) + 2
    shape = np.ceil((hi - lo) / res).astype(int)
    grid = np.zeros(shape[::-1], bool)
    ij = ((pts - lo) / res).astype(int)
    grid[ij[:, 1], ij[:, 0]] = True
    grid = ndimage.binary_closing(grid, iterations=2)
    grid = ndimage.binary_opening(grid, iterations=1)
    skel = thin(grid)
    ys, xs = np.nonzero(skel)
    cx = lo[0] + (xs + 0.5) * res
    cy = lo[1] + (ys + 0.5) * res
    cols = np.unique(xs)
    px = np.array([cx[xs == c].mean() for c in cols])
    py = np.array([cy[xs == c].mean() for c in cols])
    py = ndimage.gaussian_filter1d(ndimage.median_filter(py, size=9, mode="nearest"), 6.0, mode="nearest")
    seg = np.linalg.norm(np.diff(np.column_stack([px, py]), axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    grid_s = np.arange(0, arc[-1], step)
    line = np.column_stack([np.interp(grid_s, arc, px), np.interp(grid_s, arc, py)])
    return line, grid, lo, res


def arclength(line):
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(seg)])


def frame(line, xyz):
    s = arclength(line)
    tan = np.gradient(line, axis=0)
    tan = tan / np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.column_stack([-tan[:, 1], tan[:, 0]])
    tree = cKDTree(line)
    _, i = tree.query(xyz[:, :2])
    d = np.sum((xyz[:, :2] - line[i]) * nrm[i], axis=1)
    return s[i], d, s, nrm
