import numpy as np

from trail_hazard_map.demo.scene import C, H

TRACTION = {"trail": 0.1, "vegetation": 0.3, "rock": 0.7, "trunk": 0.6, "water": 0.9, "mud": 0.8, "snow": 0.8, "sky": 0.0}
WET_CLASSES = [C["mud"], C["water"], C["snow"]]
HAZARD_TERM_IDS = [H["log"], H["rocks"], H["dropoff"], H["branch"]]


def robust_norm(x, floor, lo_p, hi_p):
    lo, hi = np.percentile(x, [lo_p, hi_p])
    return np.clip((x - lo) / max(hi - lo, floor), 0, 1)


def gaussian_terms(cls, evidence, slope, rough):
    traction = np.array(list(TRACTION.values()))[cls]
    wet = np.isin(cls, WET_CLASSES).astype(float)
    haz = evidence[:, HAZARD_TERM_IDS].max(1)
    return {"slope": slope, "roughness": rough, "hazard": haz, "traction": traction, "wet": wet}


def segment_terms(profile, sight, cfg):
    g = cfg["geometry"]
    width = profile["width"]
    return {
        "width_deficit": np.maximum(0, np.percentile(width, 90) - width),
        "dropoff": profile["dropoff"],
        "step": profile["step"],
        "sight": g["max_sight_m"] - sight,
        "clearance": g["clearance_cap_m"] - profile["clearance"],
    }
