import numpy as np
from matplotlib import pyplot as plt

from trail_hazard_map.demo.render import render

TERM_COLORS = {
    "slope": "#8c564b",
    "roughness": "#7f7f7f",
    "width_deficit": "#17becf",
    "dropoff": "#d62728",
    "hazard": "#ff7f0e",
    "traction": "#bcbd22",
    "wet": "#1f77b4",
    "sight": "#9467bd",
    "clearance": "#2ca02c",
    "step": "#e377c2",
}


def birdseye(path, xyz, heat, line, instances, seg_m, worst):
    fig, ax = plt.subplots(figsize=(15, 4.6))
    order = np.argsort(xyz[:, 2])
    ax.scatter(xyz[order, 0], xyz[order, 1], c=heat[order], s=1.2, linewidths=0)
    ax.plot(line[:, 0], line[:, 1], color="black", lw=0.6, alpha=0.5)
    for k, inst in enumerate(instances):
        mid = 0.5 * (inst["s_start"] + inst["s_end"])
        i = min(int(mid / 0.5), len(line) - 1)
        x, y = line[i]
        ax.annotate(
            inst["type"],
            (x, y),
            xytext=(x, y + (4.5 if k % 2 == 0 else -5.0)),
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.7},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.85},
        )
    a, b = worst[0], worst[1]
    ia, ib = int(a / 0.5), min(int(b / 0.5), len(line) - 1)
    ax.plot(line[ia:ib, 0], line[ia:ib, 1], color="magenta", lw=2.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Bird's-eye hazard heatmap (green = easy, red = hazardous, gray = low confidence); magenta = worst 10 m")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def strip(path, seg_risk, mat, seg_m, worst, instances, display_max):
    nb = len(seg_risk)
    x = (np.arange(nb) + 0.5) * seg_m
    fig, ax = plt.subplots(figsize=(15, 4))
    bottom = np.zeros(nb)
    for k, v in mat.items():
        ax.bar(x, v, width=seg_m * 0.92, bottom=bottom, color=TERM_COLORS.get(k, "gray"), label=k)
        bottom += v
    ax.axvspan(worst[0], worst[1], color="magenta", alpha=0.12, lw=0)
    for inst in instances:
        ax.axvline(0.5 * (inst["s_start"] + inst["s_end"]), color="black", lw=0.6, ls=":")
        ax.text(0.5 * (inst["s_start"] + inst["s_end"]), ax.get_ylim()[1] * 0.98, inst["type"], rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("distance along trail (m)")
    ax.set_ylabel("risk contribution")
    ax.set_title("Risk vs distance, stacked by explanatory term (top-risk Gaussians per 2 m segment)")
    ax.legend(ncol=5, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def frame_montage(path, rows, labels):
    fig, axes = plt.subplots(len(rows), 3, figsize=(11, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (rgb, heat, blend) in enumerate(rows):
        for c, (img, title) in enumerate([(rgb, "RGB"), (heat, "risk"), (blend, "overlay")]):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(title)
        axes[r, 0].text(4, 14, labels[r], color="white", fontsize=9, bbox={"fc": "black", "alpha": 0.6, "pad": 2})
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def render_pair(xyz, rgb, heat, K, R, t, size, alpha=0.6):
    a, _ = render(xyz, rgb, K, R, t, size)
    b, _ = render(xyz, heat, K, R, t, size)
    return a, b, (1 - alpha) * a + alpha * b
