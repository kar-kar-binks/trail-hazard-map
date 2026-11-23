from dataclasses import dataclass

import numpy as np

CLASSES = ["trail", "vegetation", "rock", "trunk", "water", "mud", "snow", "sky"]
HAZARDS = ["log", "rocks", "mud", "dropoff", "branch"]
C = {name: i for i, name in enumerate(CLASSES)}
H = {name: i for i, name in enumerate(HAZARDS)}


@dataclass
class Splat:
    xyz: np.ndarray
    rgb: np.ndarray
    opacity: np.ndarray
    cls: np.ndarray
    hazard: np.ndarray


def path_xy(s):
    s = np.asarray(s, dtype=float)
    return np.stack([s, 7.0 * np.sin(s / 22.0)], axis=-1)


def path_tangent(s):
    s = np.asarray(s, dtype=float)
    t = np.stack([np.ones_like(s), 7.0 / 22.0 * np.cos(s / 22.0)], axis=-1)
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


def path_normal(s):
    t = path_tangent(s)
    return np.stack([-t[..., 1], t[..., 0]], axis=-1)


def width_at(s, base):
    s = np.asarray(s, dtype=float)
    return base - 1.2 * np.exp(-((s - 90.0) / 4.0) ** 2)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def terrain(x, y, s):
    z = 0.035 * s + 2.2 * smoothstep((s - 118.0) / 8.0)
    z = z + 0.12 * np.sin(x * 0.9) * np.cos(y * 0.7) + 0.08 * np.sin(x * 2.1 + y * 1.3)
    return z


def build_scene(cfg, rng):
    length = cfg["scene"]["length_m"]
    base_w = cfg["scene"]["base_width_m"]
    n = cfg["scene"]["n_ground"]
    s = rng.uniform(0, length, n)
    d = rng.uniform(-7.0, 7.0, n)
    xy = path_xy(s) + d[:, None] * path_normal(s)
    w = width_at(s, base_w)
    z = terrain(xy[:, 0], xy[:, 1], s)
    on_trail = np.abs(d) < w / 2
    cls = np.where(on_trail, C["trail"], C["vegetation"])
    haz = np.full(n, -1)
    rgb = np.where(
        on_trail[:, None],
        np.array([0.62, 0.50, 0.36]) + rng.normal(0, 0.04, (n, 3)),
        np.array([0.16, 0.42, 0.15]) + rng.normal(0, 0.06, (n, 3)),
    )
    z = z + np.where(on_trail, rng.normal(0, 0.012, n), rng.normal(0, 0.06, n))

    in_rocks = on_trail & (s > 104) & (s < 116)
    z = z + np.where(in_rocks, np.abs(rng.normal(0, 0.16, n)) * (rng.random(n) < 0.35), 0.0)
    rocky = in_rocks & (z - terrain(xy[:, 0], xy[:, 1], s) > 0.06)
    cls[rocky] = C["rock"]
    haz[rocky] = H["rocks"]
    rgb[rocky] = 0.5 + rng.normal(0, 0.05, (rocky.sum(), 3))

    in_mud = on_trail & (s > 138) & (s < 150)
    cls[in_mud] = C["mud"]
    haz[in_mud] = H["mud"]
    rgb[in_mud] = np.array([0.25, 0.17, 0.11]) + rng.normal(0, 0.03, (in_mud.sum(), 3))

    side = d > w / 2
    in_drop = (s > 165) & (s < 185) & side
    depth = 4.0 * smoothstep((d - w / 2) / 1.5) * smoothstep((s - 165) / 3) * smoothstep((185 - s) / 3)
    z = z - np.where(in_drop, depth, 0.0)
    lip = in_drop & (d < w / 2 + 1.6)
    haz[lip] = H["dropoff"]

    parts = [(xy, z, rgb, cls, haz)]

    def cylinder_across(s0, height, radius, count, color, cls_id, haz_id, extra=0.3):
        ww = float(width_at(s0, base_w))
        dd = rng.uniform(-ww / 2 - extra, ww / 2 + extra, count)
        ang = rng.uniform(0, 2 * np.pi, count)
        c = path_xy(np.full(count, s0)) + dd[:, None] * path_normal(np.full(count, s0))
        zc = terrain(c[:, 0], c[:, 1], s0) + height + radius * np.sin(ang)
        off = radius * np.cos(ang)[:, None] * path_tangent(np.full(count, s0))
        col = np.tile(np.array(color), (count, 1)) + rng.normal(0, 0.03, (count, 3))
        return (c + off, zc, col, np.full(count, cls_id), np.full(count, haz_id))

    parts.append(cylinder_across(60.0, 0.28, 0.28, 3500, [0.36, 0.25, 0.16], C["trunk"], H["log"]))
    parts.append(cylinder_across(30.0, 1.4, 0.1, 1800, [0.2, 0.36, 0.14], C["vegetation"], H["branch"], 1.0))

    for _ in range(60):
        s0 = rng.uniform(5, length - 5)
        sgn = rng.choice([-1, 1])
        ww = float(width_at(s0, base_w))
        dd = sgn * (ww / 2 + 1.0 + rng.uniform(0, 1.5))
        m = 220
        c = path_xy(np.full(m, s0)) + dd * path_normal(np.full(m, s0)) + rng.normal(0, 0.45, (m, 2))
        zb = terrain(c[:, 0], c[:, 1], s0) + rng.uniform(0.1, 2.1, m)
        col = np.array([0.12, 0.38, 0.12]) + rng.normal(0, 0.05, (m, 3))
        parts.append((c, zb, col, np.full(m, C["vegetation"]), np.full(m, -1)))

    for _ in range(30):
        s0 = rng.uniform(5, length - 5)
        sgn = rng.choice([-1, 1])
        ww = float(width_at(s0, base_w))
        dd = sgn * (ww / 2 + 2.5 + rng.uniform(0, 2))
        m = 160
        c0 = path_xy(s0) + dd * path_normal(s0)
        c = c0 + rng.normal(0, 0.12, (m, 2))
        zt = terrain(c0[0], c0[1], s0) + rng.uniform(0, 4.0, m)
        col = np.array([0.30, 0.2, 0.13]) + rng.normal(0, 0.03, (m, 3))
        parts.append((c, zt, col, np.full(m, C["trunk"]), np.full(m, -1)))

    xyz = np.concatenate([np.column_stack([p[0], p[1]]) for p in parts])
    rgb = np.clip(np.concatenate([p[2] for p in parts]), 0, 1)
    cls = np.concatenate([p[3] for p in parts]).astype(int)
    haz = np.concatenate([p[4] for p in parts]).astype(int)
    opacity = np.clip(rng.normal(0.9, 0.08, len(xyz)), 0.3, 1.0)
    return Splat(xyz, rgb, opacity, cls, haz)


def camera_path(cfg, ground_z):
    length = cfg["scene"]["length_m"]
    sp = cfg["cameras"]["spacing_m"]
    s = np.arange(2.0, length - 8.0, sp)
    pos = path_xy(s)
    look = path_xy(s + 6.0)
    z = terrain(pos[:, 0], pos[:, 1], s) + cfg["cameras"]["eye_height_m"]
    skip = (s > 70) & (s < 82)
    s, pos, look, z = s[~skip], pos[~skip], look[~skip], z[~skip]
    eye = np.column_stack([pos, z])
    tgt = np.column_stack([look, z - 0.35])
    return s, eye, tgt
