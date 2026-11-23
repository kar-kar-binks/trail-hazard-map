import base64
import json
from pathlib import Path

import numpy as np

from trail_hazard_map.demo.scene import HAZARDS


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def write_viewer(path, xyz, rgb, heat, seg_of, haz_primary, segments, rng, keep=70000):
    force = np.nonzero(haz_primary >= 0)[0]
    rest = np.nonzero(haz_primary < 0)[0]
    pick = np.concatenate([force, rng.choice(rest, min(keep, len(rest)), replace=False)])
    origin = xyz.mean(0)
    pts = xyz[pick] - origin
    segments = [{**s, "center": [float(c) for c in (np.array(s["center"]) - origin)]} for s in segments]
    data = {
        "n": int(len(pick)),
        "xyz": b64(pts.astype(np.float32)),
        "rgb": b64((rgb[pick] * 255).astype(np.uint8)),
        "heat": b64((heat[pick] * 255).astype(np.uint8)),
        "seg": b64(seg_of[pick].astype(np.int16)),
        "haz": b64(haz_primary[pick].astype(np.int8)),
        "hazards": HAZARDS,
        "segments": segments,
    }
    tpl = (Path(__file__).parent / "viewer_template.html").read_text()
    Path(path).write_text(tpl.replace("__DATA__", json.dumps(data)))
