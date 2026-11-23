import json
import time
from pathlib import Path

import numpy as np

from trail_hazard_map.classical.filters import gaussian_blur, quality_filter
from trail_hazard_map.demo.render import intrinsics, look_at, render
from trail_hazard_map.demo.scene import C, CLASSES, HAZARDS, build_scene, camera_path
from trail_hazard_map.eval.pose import pose_eval
from trail_hazard_map.eval.validity import instance_match, prf, spearman
from trail_hazard_map.geometry import centerline as cl
from trail_hazard_map.geometry.ground import local_features, ransac_plane
from trail_hazard_map.geometry.profile import ground_profile, segment_profile, sight_distance
from trail_hazard_map.scoring import combine as cb
from trail_hazard_map.scoring import explain, terms
from trail_hazard_map.semantics.fuse import fuse
from trail_hazard_map.viz import overlays
from trail_hazard_map.viz.flythrough import fly
from trail_hazard_map.viz.recolor import heat_colors, write_ply
from trail_hazard_map.viz.viewer import write_viewer


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def degrade(frames, rng):
    out = [f.copy() for f in frames]
    bad = {}
    picks = rng.choice(len(frames), 6, replace=False)
    for k, i in enumerate(picks):
        if k < 2:
            g = np.stack([gaussian_blur(out[i][..., c], 4.0) for c in range(3)], -1)
            out[i] = g
            bad[int(i)] = "blur"
        elif k < 4:
            out[i] = np.clip(out[i] * 0.12, 0, 1)
            bad[int(i)] = "dark"
        else:
            out[i] = np.clip(out[i] * 0.4 + 0.75, 0, 1)
            bad[int(i)] = "bright"
    return out, bad


def run_demo(cfg, out_dir):
    t0 = time.time()
    rng = np.random.default_rng(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seg_m = cfg["geometry"]["segment_m"]
    length = cfg["scene"]["length_m"]
    size = (cfg["cameras"]["width"], cfg["cameras"]["height"])
    K = intrinsics(cfg)

    splat = build_scene(cfg, rng)
    n = len(splat.xyz)
    log(f"synthetic splat: {n} Gaussians", t0)

    cam_s, eyes, tgts = camera_path(cfg, None)
    cams = [look_at(e, t) for e, t in zip(eyes, tgts)]
    frames = [render(splat.xyz, splat.rgb, K, R, t, size)[0] for R, t in cams]
    degraded, bad = degrade(frames, rng)
    keep, qrep = quality_filter(degraded, cfg)
    dropped = {r["frame"]: r["reason"] for r in qrep if r["reason"]}
    log(f"quality filter kept {len(keep)}/{len(frames)} frames, dropped {len(dropped)} ({dict(sorted(dropped.items()))})", t0)

    pairs = []
    for i in keep[::7][:6]:
        fwd = tgts[i] - eyes[i]
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right = right / np.linalg.norm(right)
        shift = 0.9 * fwd + 0.7 * right
        cam2 = look_at(eyes[i] + shift, tgts[i] + shift)
        img2 = render(splat.xyz, splat.rgb, K, cam2[0], cam2[1], size)[0]
        pairs.append((i, frames[i], img2, cams[i], cam2))
    pose = pose_eval(pairs, K, rng)
    log(f"two-view baseline on {len(pose)} pairs", t0)

    fz = fuse(splat, eyes, tgts, cfg, rng)
    log("multi-view semantic + hazard fusion", t0)

    cls = fz["cls"]
    trail_like = np.isin(cls, [C["trail"], C["mud"], C["rock"]])
    plane = ransac_plane(splat.xyz[cls == C["trail"]], rng)
    line, grid, lo, res = cl.extract(splat.xyz, trail_like)
    s, d, line_s, nrm = cl.frame(line, splat.xyz)
    L = float(line_s[-1])
    log(f"centerline: {len(line)} vertices, {L:.0f} m", t0)

    gz_s, gz = ground_profile(s, d, splat.xyz[:, 2], trail_like, L)
    prof = segment_profile(s, d, splat.xyz[:, 2], splat.opacity, trail_like, gz_s, gz, L, seg_m, cfg)
    sight = sight_distance(splat.xyz, splat.opacity, line, line_s, gz_s, gz, L, seg_m, cfg)
    slope, rough, planarity, _ = local_features(splat.xyz, cfg["geometry"]["knn"])
    log("geometry: slope, roughness, width, drop-off, steps, clearance, sight", t0)

    nb = len(prof["width"])
    seg_of = np.clip((s / seg_m).astype(int), 0, nb - 1)
    corridor = np.abs(d) < prof["width"][seg_of] / 2 + 0.4
    gterms = terms.gaussian_terms(cls, fz["evidence"], slope, rough)
    sterms = terms.segment_terms(prof, sight, cfg)
    normed, contrib = cb.combine(gterms, sterms, seg_of, corridor, cfg)
    risk = cb.smooth3d(cb.total(contrib), splat.xyz, cfg["scoring"]["knn_smooth"])
    seg_risk = cb.segment_scores(risk, seg_of, corridor, nb)
    worst = cb.worst_window(seg_risk, seg_m, cfg["scoring"]["worst_window_m"])
    mat = explain.segment_contrib(contrib, seg_of, corridor, risk, nb)
    tops = explain.top_terms(mat, nb)
    cam_line_s = np.array([float(line_s[np.argmin(np.linalg.norm(line - e[:2], axis=1))]) for e in eyes])
    instances = explain.hazard_instances(s, fz["evidence"], fz["verified"], cam_line_s)
    log(f"scoring: trail mean {seg_risk.mean():.3f}, worst 10 m {worst[0]:.0f}-{worst[1]:.0f} m = {worst[2]:.3f}", t0)

    focus = np.abs(d) < prof["width"][seg_of] / 2 + 1.6
    heat = heat_colors(risk, fz["views"], cfg["scoring"]["display_max"], focus)
    write_ply(out / "scene_rgb.ply", splat.xyz, splat.rgb)
    write_ply(out / "scene_heat.ply", splat.xyz, heat, risk)

    overlays.birdseye(out / "birdseye.png", splat.xyz, heat, line, instances, seg_m, worst)
    overlays.strip(out / "score_strip.png", seg_risk, mat, seg_m, worst, instances, cfg["scoring"]["display_max"])
    rows, labels = [], []
    for inst in instances[:5]:
        if not inst["evidence_frames"]:
            continue
        ci = inst["evidence_frames"][-1]
        R, t = cams[ci]
        rows.append(overlays.render_pair(splat.xyz, splat.rgb, heat, K, R, t, size))
        labels.append(f"{inst['type']} @ {inst['s_start']:.0f}-{inst['s_end']:.0f} m (frame {ci})")
    overlays.frame_montage(out / "frame_overlays.png", rows, labels)
    log("bird's-eye, strip chart, frame overlays", t0)

    centers = (np.arange(nb) + 0.5) * seg_m
    cpos = np.column_stack([np.interp(centers, line_s, line[:, 0]), np.interp(centers, line_s, line[:, 1]), prof["ground"]])
    disp = np.clip(seg_risk / cfg["scoring"]["display_max"], 0, 1)
    from matplotlib import colormaps

    segcol = (colormaps["RdYlGn_r"](disp)[:, :3] * 255).tolist()
    segments = [
        {
            "s0": float(i * seg_m),
            "risk": float(seg_risk[i]),
            "color": segcol[i],
            "top": tops[i],
            "width": float(prof["width"][i]),
            "sight": float(sight[i]),
            "clearance": float(prof["clearance"][i]),
            "center": cpos[i].tolist(),
        }
        for i in range(nb)
    ]
    hz_primary = np.where(fz["verified"].any(1), fz["verified"].argmax(1), -1)
    write_viewer(out / "viewer.html", splat.xyz, splat.rgb, heat, seg_of, hz_primary, segments, rng)
    fly(out / "flythrough.gif", splat.xyz, splat.rgb, heat, K, size, line, line_s, gz_s, gz, cfg["cameras"]["eye_height_m"])
    log("viewer.html, flythrough.gif", t0)

    truth_h = {h: splat.hazard == h for h in range(len(HAZARDS))}
    fused_pr = {HAZARDS[h]: prf(fz["verified"][:, h], truth_h[h]) for h in range(len(HAZARDS))}
    naive_pr = {HAZARDS[h]: prf(fz["naive"][:, h], truth_h[h]) for h in range(len(HAZARDS))}
    seg_x = np.interp((np.arange(nb) + 0.5) * seg_m, line_s, line[:, 0])
    rho, pval = spearman(seg_risk, seg_x)
    scene_inst = [{**i, "s_start": float(np.interp(i["s_start"], line_s, line[:, 0])), "s_end": float(np.interp(i["s_end"], line_s, line[:, 0]))} for i in instances]
    metrics = {
        "note": "synthetic scene with simulated per-view predictions; validity is against injected ground truth, not real trails",
        "gaussians": int(n),
        "frames": {"total": len(frames), "kept": len(keep), "dropped": dropped, "injected_defects": bad},
        "two_view_baseline": pose,
        "semantic_accuracy": {
            "single_view": float((fz["single_cls"] == splat.cls).mean()),
            "multi_view_fused": float((cls == splat.cls).mean()),
        },
        "hazard_gaussian_pr": {"single_view_naive": naive_pr, "fused_verified": fused_pr},
        "hazard_instances": instance_match(scene_inst),
        "ground_plane_normal": [float(v) for v in plane[0]],
        "trail_length_m": L,
        "trail_mean_risk": float(seg_risk.mean()),
        "trail_p90_risk": float(np.percentile(seg_risk, 90)),
        "worst_10m": {"start_m": worst[0], "end_m": worst[1], "risk": worst[2]},
        "spearman_vs_injected_difficulty": {"rho": rho, "p": pval},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "hazards.json").write_text(json.dumps(instances, indent=2))
    (out / "segments.json").write_text(
        json.dumps([{"s0": s["s0"], "risk": s["risk"], "top_terms": s["top"], "width_m": s["width"], "sight_m": s["sight"], "clearance_m": s["clearance"]} for s in segments], indent=2)
    )
    write_report(out / "report.md", metrics, instances, tops, seg_risk, seg_m)
    log(f"done -> {out}", t0)
    return metrics


def write_report(path, m, instances, tops, seg_risk, seg_m):
    w = m["worst_10m"]
    lines = [
        "# trail-hazard-map demo report",
        "",
        "Synthetic scene; classical CV modules (blur/exposure filter, Harris, ratio-test matching, 8-point RANSAC) are real and from scratch. VGGT, gsplat and Grounded-SAM-2 are replaced by a procedural splat and simulated per-view predictions.",
        "",
        f"- Trail length: {m['trail_length_m']:.0f} m",
        f"- Mean risk: {m['trail_mean_risk']:.3f}, 90th percentile segment: {m['trail_p90_risk']:.3f}",
        f"- Worst 10 m: {w['start_m']:.0f}-{w['end_m']:.0f} m (risk {w['risk']:.3f})",
        f"- Spearman vs injected difficulty: {m['spearman_vs_injected_difficulty']['rho']:.2f}",
        f"- Semantic accuracy: single view {m['semantic_accuracy']['single_view']:.3f}, fused {m['semantic_accuracy']['multi_view_fused']:.3f}",
        "",
        "## Detected hazards",
        "",
        "| type | span (m) | confidence | evidence frames |",
        "|---|---|---|---|",
    ]
    for i in instances:
        lines.append(f"| {i['type']} | {i['s_start']:.0f}-{i['s_end']:.0f} | {i['confidence']:.2f} | {i['evidence_frames']} |")
    lines += ["", "## Ten riskiest segments", "", "| span (m) | risk | top terms |", "|---|---|---|"]
    for i in np.argsort(-seg_risk)[:10]:
        t = ", ".join(f"{x['term']} {x['contribution']:.3f}" for x in tops[i])
        lines.append(f"| {i * seg_m:.0f}-{(i + 1) * seg_m:.0f} | {seg_risk[i]:.3f} | {t} |")
    path.write_text("\n".join(lines) + "\n")
