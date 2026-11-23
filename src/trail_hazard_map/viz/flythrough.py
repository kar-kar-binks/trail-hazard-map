import numpy as np
from PIL import Image

from trail_hazard_map.demo.render import look_at, render


def fly(path, xyz, rgb, heat, K, size, line, line_s, gz_s, gz, eye_h, frames=120, alpha=0.65):
    out = []
    raw = []
    xy = []

    def pos(q):
        return np.array([np.interp(q, line_s, line[:, 0]), np.interp(q, line_s, line[:, 1]), np.interp(q, gz_s, gz)])

    ss = np.linspace(1.0, line_s[-1] - 9.0, frames)
    for s in ss:
        eye = pos(s) + np.array([0, 0, eye_h])
        tgt = pos(s + 7.0) + np.array([0, 0, eye_h - 0.5])
        R, t = look_at(eye, tgt)
        a, _ = render(xyz, rgb, K, R, t, size)
        b, _ = render(xyz, heat, K, R, t, size)
        img = (1 - alpha) * a + alpha * b
        out.append(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)))
        raw.append(np.clip(img, 0, 1))
        xy.append(eye[:2])
    out[0].save(path, save_all=True, append_images=out[1:], duration=70, loop=0, optimize=True)
    return raw, xy
