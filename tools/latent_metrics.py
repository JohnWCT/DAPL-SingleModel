import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist


def sanitize_latent(latent, name="latent"):
    arr = np.asarray(latent, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D latent matrix, got shape {arr.shape}")
    finite_mask = np.isfinite(arr).all(axis=1)
    return arr[finite_mask]


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def calculate_fid(source_latent, target_latent):
    source = sanitize_latent(source_latent, name="source_latent")
    target = sanitize_latent(target_latent, name="target_latent")
    if source.shape[0] < 2 or target.shape[0] < 2:
        return float("inf")
    mu1 = np.mean(source, axis=0)
    sigma1 = np.cov(source, rowvar=False)
    mu2 = np.mean(target, axis=0)
    sigma2 = np.cov(target, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def calculate_mmd(source_latent, target_latent, max_samples=1000, gamma=None, random_seed=42):
    source = sanitize_latent(source_latent, name="source_latent")
    target = sanitize_latent(target_latent, name="target_latent")
    rng = np.random.default_rng(random_seed)
    if len(source) > max_samples:
        source = source[rng.choice(len(source), max_samples, replace=False)]
    if len(target) > max_samples:
        target = target[rng.choice(len(target), max_samples, replace=False)]
    if gamma is None:
        gamma = 1.0 / max(1, source.shape[1])
    xx = np.exp(-gamma * cdist(source, source, "sqeuclidean"))
    yy = np.exp(-gamma * cdist(target, target, "sqeuclidean"))
    xy = np.exp(-gamma * cdist(source, target, "sqeuclidean"))
    return float(max(0.0, xx.mean() + yy.mean() - 2 * xy.mean()))


def calculate_wasserstein(source_latent, target_latent):
    source = sanitize_latent(source_latent, name="source_latent")
    target = sanitize_latent(target_latent, name="target_latent")
    return float(np.linalg.norm(np.mean(source, axis=0) - np.mean(target, axis=0)))
