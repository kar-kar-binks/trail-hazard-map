import json
import math
import sys
from collections import defaultdict
from pathlib import Path

AREAS = {
    "lincoln_woods": "Lincoln Woods State Park, Lincoln",
    "neutaconkanut": "Neutaconkanut Hill, Providence",
    "snake_den": "Snake Den State Park, Johnston",
    "diamond_hill": "Diamond Hill Park, Cumberland",
    "blackstone_gorge": "Blackstone River Gorge, Blackstone",
    "goddard": "Goddard Memorial State Park, Warwick",
}


def length_m(pts):
    return sum(math.hypot((b[0] - a[0]) * 82900, (b[1] - a[1]) * 111200) for a, b in zip(pts, pts[1:]))


def key(p):
    return (round(p[0], 5), round(p[1], 5))


def chain(ways):
    pool = [list(w) for w in ways]
    best = []
    for start in range(len(pool)):
        cur = list(pool[start])
        rest = [w for i, w in enumerate(pool) if i != start]
        grown = True
        while grown:
            grown = False
            for w in rest:
                if key(w[0]) == key(cur[-1]):
                    cur += w[1:]
                elif key(w[-1]) == key(cur[-1]):
                    cur += w[::-1][1:]
                elif key(w[-1]) == key(cur[0]):
                    cur = w[:-1] + cur
                elif key(w[0]) == key(cur[0]):
                    cur = w[::-1][:-1] + cur
                else:
                    continue
                rest.remove(w)
                grown = True
                break
        if length_m(cur) > length_m(best):
            best = cur
    return best


def build(raw_dir, out_path, min_len=450, per_area=9):
    trails = []
    for aid, label in AREAS.items():
        f = Path(raw_dir) / f"{aid}.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        by_name = defaultdict(list)
        for e in data.get("elements", []):
            geom = [[g["lon"], g["lat"]] for g in e.get("geometry", [])]
            if len(geom) > 1:
                by_name[e["tags"]["name"]].append(geom)
        found = []
        for name, ways in by_name.items():
            c = chain(ways)
            L = length_m(c)
            if L >= min_len:
                found.append((L, name, c))
        for L, name, c in sorted(found, reverse=True)[:per_area]:
            trails.append({"id": f"{aid}:{name}", "area": label, "name": name, "length_m": round(L), "coords": [[round(x, 5), round(y, 5)] for x, y in c]})
    Path(out_path).write_text("window.TRAILS = " + json.dumps(trails, separators=(",", ":")) + ";\n")
    return trails


if __name__ == "__main__":
    t = build(sys.argv[1], sys.argv[2])
    for x in t:
        print(x["area"], "|", x["name"], x["length_m"], len(x["coords"]))
