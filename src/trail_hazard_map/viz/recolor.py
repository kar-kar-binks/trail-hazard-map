import numpy as np
from matplotlib import colormaps


def heat_colors(risk, views, display_max, focus, min_views=10):
    x = np.clip(risk / display_max, 0, 1)
    col = colormaps["RdYlGn_r"](x)[:, :3]
    conf = np.clip(views / min_views, 0, 1)
    mix = (0.65 * (1 - conf))[:, None]
    col = col * (1 - mix) + 0.55 * mix
    off = (0.35 * col + 0.65 * 0.35)
    return np.where(focus[:, None], col, off)


def write_ply(path, xyz, rgb, extra=None):
    n = len(xyz)
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if extra is not None:
        props.append(("risk", "<f4"))
    arr = np.empty(n, dtype=props)
    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    c = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    arr["red"], arr["green"], arr["blue"] = c[:, 0], c[:, 1], c[:, 2]
    if extra is not None:
        arr["risk"] = extra
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    for name, dt in props:
        header.append(f"property {'float' if dt == '<f4' else 'uchar'} {name}")
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())
