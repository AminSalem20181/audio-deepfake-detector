from __future__ import annotations
import torch
from scipy.stats import spearmanr
from .gradcam import gradcam

def cam_stability(model, x: torch.Tensor, sigmas=(0.05, 0.1, 0.2)):
    model.eval(); out = {}
    clean = gradcam(model, x)
    for sigma in sigmas:
        noisy = gradcam(model, x + torch.randn_like(x) * sigma)
        diffs=[]; cors=[]
        for a,b in zip(clean.flatten(1).cpu().numpy(), noisy.flatten(1).cpu().numpy()):
            diffs.append(float(abs(a-b).mean())); cors.append(float(spearmanr(a,b).statistic))
        out[float(sigma)] = {'mean_abs_difference': sum(diffs)/len(diffs), 'spearman': sum(cors)/len(cors)}
    return out
