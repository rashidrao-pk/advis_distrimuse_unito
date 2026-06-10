from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def anomaly_map(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - recon).mean(dim=1)


def anomaly_score(
    x: torch.Tensor,
    recon: torch.Tensor,
    method: str = "max",
    quantile: float = 1.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    amap = anomaly_map(x, recon)
    flat = amap.flatten(start_dim=1)
    if method == "max":
        return flat.max(dim=1).values
    if method == "mean":
        return flat.mean(dim=1)
    if method == "quantile":
        return torch.quantile(flat, q=quantile, dim=1)
    if method in {"mean_plus_sigma", "mean+n_sigma"}:
        return flat.mean(dim=1) + sigma * flat.std(dim=1)
    raise ValueError(f"Unknown anomaly score method: {method}")


def score_np_pair(
    image: np.ndarray,
    reconstruction: np.ndarray,
    method: str = "max",
    quantile: float = 1.0,
    sigma: float = 1.0,
    gaussian_sigma: float = 0.0,
) -> float:
    diff = np.abs(image.astype(np.float32) - reconstruction.astype(np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    if gaussian_sigma > 0:
        diff = gaussian_filter(diff, sigma=gaussian_sigma)
    flat = diff.reshape(-1)
    if method == "max":
        return float(flat.max())
    if method == "mean":
        return float(flat.mean())
    if method == "quantile":
        return float(np.quantile(flat, quantile))
    if method in {"mean_plus_sigma", "mean+n_sigma"}:
        return float(flat.mean() + sigma * flat.std())
    raise ValueError(f"Unknown anomaly score method: {method}")
