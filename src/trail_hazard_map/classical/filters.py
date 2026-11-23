import numpy as np
from scipy import signal


def to_gray(img):
    return img @ np.array([0.299, 0.587, 0.114])


def convolve(img, kernel):
    return signal.convolve2d(img, kernel, mode="same", boundary="symm")


def gaussian_kernel(sigma):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return k


def gaussian_blur(img, sigma):
    k = gaussian_kernel(sigma)
    tmp = signal.convolve2d(img, k[None, :], mode="same", boundary="symm")
    return signal.convolve2d(tmp, k[:, None], mode="same", boundary="symm")


def laplacian_variance(gray):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
    return float(convolve(gray * 255.0, k).var())


def exposure_stats(gray):
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    p = hist / hist.sum()
    return {
        "mean": float(gray.mean() * 255.0),
        "contrast": float(gray.std() * 255.0),
        "clipped_dark": float(p[:2].sum()),
        "clipped_bright": float(p[-2:].sum()),
    }


def gaussian_pyramid(gray, levels=3, sigma=1.0):
    out = [gray]
    for _ in range(levels - 1):
        out.append(gaussian_blur(out[-1], sigma)[::2, ::2])
    return out


def quality_filter(frames, cfg):
    q = cfg["quality"]
    keep, report = [], []
    for i, f in enumerate(frames):
        g = to_gray(f)
        blur = laplacian_variance(g)
        ex = exposure_stats(g)
        reason = None
        if blur < q["min_blur_var"]:
            reason = "blurry"
        elif ex["mean"] < q["min_mean"]:
            reason = "underexposed"
        elif ex["mean"] > q["max_mean"]:
            reason = "overexposed"
        report.append({"frame": i, "blur_var": blur, "mean": ex["mean"], "reason": reason})
        if reason is None:
            keep.append(i)
    return keep, report
