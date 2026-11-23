import numpy as np

from trail_hazard_map.classical.filters import gaussian_pyramid, laplacian_variance
from trail_hazard_map.classical.ransac import eight_point, ransac_fundamental, sampson


def test_blur_lowers_laplacian_variance():
    rng = np.random.default_rng(0)
    sharp = rng.random((64, 64))
    soft = (sharp + np.roll(sharp, 1, 0) + np.roll(sharp, 1, 1) + np.roll(sharp, (1, 1), (0, 1))) / 4
    assert laplacian_variance(soft) < laplacian_variance(sharp)


def test_pyramid_halves_size():
    levels = gaussian_pyramid(np.zeros((64, 48)), 3)
    assert [l.shape for l in levels] == [(64, 48), (32, 24), (16, 12)]


def test_ransac_recovers_fundamental_with_outliers():
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.uniform(-2, 2, 120), rng.uniform(-1, 1, 120), rng.uniform(4, 9, 120)])
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1.0]])
    t = np.array([0.5, 0.1, 0.2])
    def proj(P):
        h = P @ K.T
        return h[:, :2] / h[:, 2:]
    p1, p2 = proj(X), proj(X + t)
    p2[:20] += rng.uniform(-40, 40, (20, 2))
    F, inl = ransac_fundamental(p1, p2, rng)
    assert np.sqrt(np.median(sampson(F, p1[20:], p2[20:]))) < 1.0
    assert inl[20:].mean() > 0.8
    assert np.linalg.matrix_rank(eight_point(p1[20:], p2[20:])) == 2
