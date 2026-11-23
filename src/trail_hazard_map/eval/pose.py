import numpy as np

from trail_hazard_map.classical.features import describe, harris, match_ratio
from trail_hazard_map.classical.filters import to_gray
from trail_hazard_map.classical.ransac import fundamental_from_poses, ransac_fundamental, sampson


def two_view_baseline(img1, img2, F_ref, rng):
    g1, g2 = to_gray(img1), to_gray(img2)
    k1, d1 = describe(g1, harris(g1))
    k2, d2 = describe(g2, harris(g2))
    m = match_ratio(d1, d2)
    p1, p2 = k1[m[:, 0]], k2[m[:, 1]]
    if len(p1) < 12:
        return None
    F, inl = ransac_fundamental(p1, p2, rng)
    err_est = float(np.sqrt(np.median(sampson(F, p1[inl], p2[inl]))))
    err_ref = float(np.sqrt(np.median(sampson(F_ref, p1[inl], p2[inl]))))
    return {"matches": int(len(p1)), "inliers": int(inl.sum()), "err_estimated_px": err_est, "err_reference_px": err_ref}


def pose_eval(pairs, K, rng):
    rows = []
    for tag, img1, img2, cam1, cam2 in pairs:
        F_ref = fundamental_from_poses(K, cam1[0], cam1[1], cam2[0], cam2[1])
        r = two_view_baseline(img1, img2, F_ref, rng)
        if r:
            rows.append({"frame": int(tag), **r})
    return rows
