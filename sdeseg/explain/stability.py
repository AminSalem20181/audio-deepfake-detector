"""CAM perturbation stability; constant maps have undefined rank correlation."""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import spearmanr
from .gradcam import gradcam


def cam_stability(model, x: torch.Tensor, sigmas=(0.05, 0.1, 0.2)):
    model.eval()
    clean = gradcam(model, x)
    result = {}
    for sigma in sigmas:
        noisy = gradcam(model, x + torch.randn_like(x) * sigma)
        correlations = []
        for a, b in zip(clean.flatten(1).cpu().numpy(), noisy.flatten(1).cpu().numpy()):
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                correlations.append(float(spearmanr(a, b).statistic))
        result[float(sigma)] = dict(mean_abs_difference=float((clean-noisy).abs().mean()),
            spearman=float(np.mean(correlations)) if correlations else None,
            valid_correlations=len(correlations), count=len(x))
    return result
