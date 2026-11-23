import numpy as np
from scipy.spatial import cKDTree

from trail_hazard_map.demo.scene import CLASSES, HAZARDS


def fuse(splat, eyes, targets, cfg, rng):
    f = cfg["fusion"]
    k = len(CLASSES)
    n = len(splat.xyz)
    tree = cKDTree(splat.xyz)
    score = np.zeros((n, k))
    views = np.zeros(n, int)
    votes = np.zeros((n, len(HAZARDS)), int)
    single = np.full(n, -1)
    for eye, tgt in zip(eyes, targets):
        fwd = (tgt - eye) / np.linalg.norm(tgt - eye)
        idx = np.array(tree.query_ball_point(eye, f["radius_m"]), dtype=int)
        rel = splat.xyz[idx] - eye
        dist = np.linalg.norm(rel, axis=1)
        vis = (rel @ fwd) > 0.3 * dist
        idx, dist = idx[vis], dist[vis]
        if len(idx) == 0:
            continue
        p = f["p_correct_near"] + (f["p_correct_far"] - f["p_correct_near"]) * np.clip(dist / f["radius_m"], 0, 1)
        truth = splat.cls[idx]
        pred = np.where(rng.random(len(idx)) < p, truth, rng.integers(0, k, len(idx)))
        weight = np.log(p * (k - 1) / (1 - p))
        np.add.at(score, (idx, pred), weight)
        views[idx] += 1
        fresh = single[idx] < 0
        single[idx[fresh]] = pred[fresh]
        for h in range(len(HAZARDS)):
            is_h = splat.hazard[idx] == h
            det = np.where(is_h, rng.random(len(idx)) < f["p_hazard_detect"], rng.random(len(idx)) < f["p_hazard_false"])
            votes[idx[det], h] += 1
    fused = np.where(views > 0, score.argmax(1), C_DEFAULT)
    frac = votes / np.maximum(views, 1)[:, None]
    verified = (votes >= f["min_views"]) & (frac >= f["min_fraction"])
    evidence = np.where(verified, frac, 0.0)
    naive = votes >= 1
    return {
        "cls": fused,
        "single_cls": np.where(single >= 0, single, C_DEFAULT),
        "views": views,
        "votes": votes,
        "evidence": evidence,
        "verified": verified,
        "naive": naive,
    }


C_DEFAULT = CLASSES.index("vegetation")
