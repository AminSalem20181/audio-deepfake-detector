from __future__ import annotations
import torch
import torch.nn.functional as F

def gradcam(model, x: torch.Tensor, feature: str = "fp") -> torch.Tensor:
    model.zero_grad(set_to_none=True); logit, f, a, fp = model(x, return_maps=True); target = fp if feature == "fp" else f
    grad = torch.autograd.grad(logit.sum(), target, retain_graph=False)[0]
    weights = grad.mean((2,3), keepdim=True); cam=F.relu((weights*target).sum(1)); cam=F.interpolate(cam[:,None], (64,400), mode='bilinear', align_corners=False)[:,0]
    lo, hi=cam.flatten(1).min(1)[0][:,None,None], cam.flatten(1).max(1)[0][:,None,None]; return ((cam-lo)/(hi-lo).clamp_min(1e-8)).clamp(0,1)
