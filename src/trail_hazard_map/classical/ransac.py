import numpy as np


def normalize(pts):
    c = pts.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    h = np.column_stack([pts, np.ones(len(pts))]) @ T.T
    return h, T


def eight_point(p1, p2):
    h1, T1 = normalize(p1)
    h2, T2 = normalize(p2)
    A = np.stack(
        [
            h2[:, 0] * h1[:, 0],
            h2[:, 0] * h1[:, 1],
            h2[:, 0],
            h2[:, 1] * h1[:, 0],
            h2[:, 1] * h1[:, 1],
            h2[:, 1],
            h1[:, 0],
            h1[:, 1],
            np.ones(len(h1)),
        ],
        1,
    )
    _, _, vt = np.linalg.svd(A)
    F = vt[-1].reshape(3, 3)
    u, s, vt = np.linalg.svd(F)
    F = u @ np.diag([s[0], s[1], 0]) @ vt
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def sampson(F, p1, p2):
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])
    Fx1 = x1 @ F.T
    Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def ransac_fundamental(p1, p2, rng, iters=2000, thresh=1.5):
    n = len(p1)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 8, replace=False)
        F = eight_point(p1[idx], p2[idx])
        inl = sampson(F, p1, p2) < thresh ** 2
        if inl.sum() > best.sum():
            best = inl
    F = eight_point(p1[best], p2[best])
    return F, best


def fundamental_from_poses(K, R1, t1, R2, t2):
    R = R2 @ R1.T
    t = t2 - R @ t1
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R
    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki
    return F / np.linalg.norm(F)
