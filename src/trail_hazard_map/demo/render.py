import numpy as np
from scipy import ndimage


def look_at(eye, target):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 0.0, 1.0]))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])
    return R, -R @ eye


def intrinsics(cfg):
    c = cfg["cameras"]
    return np.array([[c["focal_px"], 0, c["width"] / 2], [0, c["focal_px"], c["height"] / 2], [0, 0, 1.0]])


def project(xyz, K, R, t):
    cam = xyz @ R.T + t
    z = cam[:, 2]
    uv = cam @ K.T
    uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
    return uv, z


def render(xyz, colors, K, R, t, size):
    w, h = size
    uv, z = project(xyz, K, R, t)
    ok = (z > 0.5) & (uv[:, 0] >= 1) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 2)
    idx = np.nonzero(ok)[0]
    order = idx[np.argsort(-z[idx])]
    u = np.round(uv[order, 0]).astype(int)
    v = np.round(uv[order, 1]).astype(int)
    col = colors[order]
    near = z[order] < 14.0
    img = np.zeros((h, w, 3))
    depth = np.full((h, w), np.inf)
    filled = np.zeros((h, w), bool)
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
        sel = np.ones(len(order), bool) if (dx, dy) == (0, 0) else near
        uu, vv = u[sel] + dx, v[sel] + dy
        img[vv, uu] = col[sel]
        depth[vv, uu] = z[order][sel]
        filled[vv, uu] = True
    dist, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
    fill = (~filled) & (dist < 3.5)
    img[fill] = img[iy[fill], ix[fill]]
    depth[fill] = depth[iy[fill], ix[fill]]
    sky = ~(filled | fill)
    grad = np.linspace(0.55, 0.85, h)[:, None]
    skycol = np.stack([grad * 0.8, grad * 0.9, np.ones_like(grad)], -1) * np.ones((1, w, 1))
    img[sky] = skycol[sky]
    return np.clip(img, 0, 1), depth
