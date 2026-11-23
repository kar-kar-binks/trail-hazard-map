import numpy as np
from scipy.stats import spearmanr

TRUTH_SPANS = [
    ("branch", 28, 32, 0.6),
    ("log", 58, 62, 1.0),
    ("narrow", 86, 94, 0.5),
    ("rocks", 104, 116, 0.7),
    ("steep", 118, 126, 0.6),
    ("mud", 138, 150, 0.8),
    ("dropoff", 165, 185, 1.0),
]
TRUTH_HAZARD = {"branch": "branch", "log": "log", "rocks": "rocks", "mud": "mud", "dropoff": "dropoff"}


def truth_difficulty(centers):
    out = np.zeros(len(centers))
    for _, a, b, w in TRUTH_SPANS:
        out[(centers >= a) & (centers <= b)] += w
    return out


def spearman(seg_risk, centers_x):
    truth = truth_difficulty(centers_x)
    rho, p = spearmanr(seg_risk, truth)
    return float(rho), float(p)


def instance_match(instances, tol=5.0):
    spans = {n: (a, b) for n, a, b, _ in TRUTH_SPANS if n in TRUTH_HAZARD}
    hit, fp = 0, 0
    for n, (a, b) in spans.items():
        got = [i for i in instances if i["type"] == TRUTH_HAZARD[n] and i["s_end"] >= a - tol and i["s_start"] <= b + tol]
        hit += bool(got)
    for i in instances:
        ok = any(TRUTH_HAZARD[n] == i["type"] and i["s_end"] >= a - tol and i["s_start"] <= b + tol for n, (a, b) in spans.items())
        fp += not ok
    return {"recall": hit / len(spans), "true_instances": len(spans), "found": len(instances), "false_instances": fp}


def prf(pred, truth):
    tp = float((pred & truth).sum())
    p = tp / max(pred.sum(), 1)
    r = tp / max(truth.sum(), 1)
    return {"precision": p, "recall": r}
