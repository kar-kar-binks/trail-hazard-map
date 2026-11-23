import json
import subprocess
import shutil
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy import ndimage

CMAP = "jet"


def risk_grid(xyz, risk, focus, res=0.5, smooth=0.8):
    lo = xyz[:, :2].min(0)
    hi = xyz[:, :2].max(0)
    shape = np.ceil((hi - lo) / res).astype(int) + 1
    ij = ((xyz[focus, :2] - lo) / res).astype(int)
    total = np.zeros(shape[::-1])
    count = np.zeros(shape[::-1])
    np.add.at(total, (ij[:, 1], ij[:, 0]), risk[focus])
    np.add.at(count, (ij[:, 1], ij[:, 0]), 1)
    z = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    z = ndimage.gaussian_filter(z, smooth)
    xs = lo[0] + np.arange(shape[0]) * res
    ys = lo[1] + np.arange(shape[1]) * res
    X, Y = np.meshgrid(xs, ys)
    return X, Y, z


def write_plotly(path, X, Y, Z, zmax):
    step = 1
    data = {
        "x": np.round(X[0, ::step], 2).tolist(),
        "y": np.round(Y[::step, 0], 2).tolist(),
        "z": np.round(Z[::step, ::step], 3).tolist(),
        "zmax": float(zmax),
    }
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trail risk surface</title>
<style>html,body{margin:0;height:100%;background:#000}#p{width:100vw;height:100vh}</style>
</head>
<body>
<div id="p"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"></script>
<script>
const D = __DATA__;
const axis = t => ({title: t, color: "#ddd", gridcolor: "#555", backgroundcolor: "#000", showbackground: true, zerolinecolor: "#777"});
Plotly.newPlot("p", [{
  type: "surface", x: D.x, y: D.y, z: D.z, surfacecolor: D.z, colorscale: "Jet", cmin: 0, cmax: D.zmax,
  colorbar: {title: {text: "risk", font: {color: "#fff"}}, tickfont: {color: "#fff"}, len: 0.7},
  lighting: {ambient: 0.75, diffuse: 0.6, specular: 0.15, roughness: 0.6}
}], {
  paper_bgcolor: "#000",
  margin: {l: 0, r: 0, t: 0, b: 0},
  scene: {
    xaxis: axis("x (m)"), yaxis: axis("y (m)"), zaxis: axis("risk score"),
    aspectmode: "manual", aspectratio: {x: 4.2, y: 1, z: 0.9},
    camera: {eye: {x: 0.9, y: -2.4, z: 1.3}}
  }
}, {responsive: true});
</script>
</body>
</html>
"""
    Path(path).write_text(html.replace("__DATA__", json.dumps(data)))


def draw_frame(fig_path, flight_img, X, Y, Z, zmax, pos, azim, elev=27, ground=None):
    fig = plt.figure(figsize=(13, 6.2), facecolor="black")
    ax = fig.add_axes([0.0, -0.08, 0.93, 1.12], projection="3d", facecolor="black")
    norm = Normalize(0, zmax)
    ax.plot_surface(X, Y, Z, facecolors=colormaps[CMAP](norm(Z)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((3.4, 1, 1.0), zoom=1.55)
    ax.view_init(elev=elev, azim=azim)
    a0 = fig.add_axes([0.015, 0.60, 0.30, 0.37])
    a0.imshow(flight_img)
    a0.set_xticks([])
    a0.set_yticks([])
    for sp in a0.spines.values():
        sp.set_edgecolor("white")
    a0.set_title("camera view with risk overlay", color="white", fontsize=9, pad=3)
    ax.set_zlim(0, zmax)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 1))
        axis.label.set_color("white")
        axis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.6)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("risk score", fontsize=8)
    if pos is not None:
        ax.scatter([pos[0]], [pos[1]], [pos[2]], c="white", s=70, edgecolors="black", depthshade=False)
    cax = fig.add_axes([0.955, 0.2, 0.014, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("risk", color="white")
    fig.savefig(fig_path, dpi=100, facecolor="black")
    plt.close(fig)


def write_demo_video(out_dir, frames, xy, X, Y, Z, zmax, fps=20):
    out_dir = Path(out_dir)
    tmp = out_dir / "_frames"
    tmp.mkdir(exist_ok=True)
    n = len(frames)
    azims = -62 + 22 * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    for k in range(n):
        ci = int(np.clip(round((xy[k][0] - X[0, 0]) / (X[0, 1] - X[0, 0])), 0, Z.shape[1] - 1))
        cj = int(np.clip(round((xy[k][1] - Y[0, 0]) / (Y[1, 0] - Y[0, 0])), 0, Z.shape[0] - 1))
        pos = (xy[k][0], xy[k][1], Z[cj, ci] + 0.03)
        draw_frame(tmp / f"f{k:04d}.png", frames[k], X, Y, Z, zmax, pos, azims[k])
    mp4 = out_dir / "demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(mp4)], check=True)
    gif = out_dir / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%04d.png"), "-vf", "fps=12,scale=880:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)], check=True)
    draw_frame(out_dir / "surface.png", frames[n // 2], X, Y, Z, zmax, None, -62)
    shutil.rmtree(tmp)
    return mp4, gif
