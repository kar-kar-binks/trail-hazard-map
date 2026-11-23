import numpy as np
from scipy.spatial import cKDTree


def ransac_plane(points, rng, iters=300, thresh=0.15):
    best, best_n = None, -1
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ s[0]
        inl = np.abs(points @ n + d) < thresh
        if inl.sum() > best_n:
            best, best_n = (n, d), inl.sum()
    return best


def local_features(xyz, k):
    tree = cKDTree(xyz)
    _, nbr = tree.query(xyz, k=k)
    pts = xyz[nbr]
    pts = pts - pts.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", pts, pts) / k
    w, v = np.linalg.eigh(cov)
    normal = v[:, :, 0]
    slope = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
    rough = np.sqrt(np.maximum(w[:, 0], 0))
    planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return slope, rough, planarity, tree
