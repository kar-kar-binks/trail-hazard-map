import numpy as np

from trail_hazard_map.demo.scene import HAZARDS


def segment_contrib(contrib, seg_of, corridor, risk, nb):
    names = list(contrib)
    mat = {k: np.zeros(nb) for k in names}
    for i in range(nb):
        sel = np.nonzero(corridor & (seg_of == i))[0]
        if len(sel) == 0:
            continue
        hot = sel[risk[sel] >= np.percentile(risk[sel], 85)]
        for k in names:
            mat[k][i] = contrib[k][hot].mean()
    return mat


def top_terms(mat, nb, top=3):
    out = []
    for i in range(nb):
        ranked = sorted(((k, float(v[i])) for k, v in mat.items()), key=lambda kv: -kv[1])[:top]
        out.append([{"term": k, "contribution": round(v, 4)} for k, v in ranked if v > 1e-4])
    return out


def hazard_instances(s, evidence, verified, cam_s, gap=4.0, min_gaussians=40):
    found = []
    for h, name in enumerate(HAZARDS):
        idx = np.nonzero(verified[:, h])[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(s[idx])]
        groups = np.split(order, np.nonzero(np.diff(s[order]) > gap)[0] + 1)
        for g in groups:
            if len(g) < min_gaussians:
                continue
            s0, s1 = float(s[g].min()), float(s[g].max())
            mid = 0.5 * (s0 + s1)
            behind = np.nonzero((cam_s < mid) & (cam_s > mid - 14))[0]
            found.append(
                {
                    "type": name,
                    "s_start": round(s0, 1),
                    "s_end": round(s1, 1),
                    "gaussians": int(len(g)),
                    "confidence": round(float(evidence[g, h].mean()), 3),
                    "evidence_frames": [int(i) for i in behind[-2:]],
                }
            )
    return sorted(found, key=lambda r: r["s_start"])
