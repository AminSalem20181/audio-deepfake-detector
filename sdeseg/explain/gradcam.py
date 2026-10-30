"""Spoof-logit Grad-CAM at F' (or F when SEGM is absent)."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def gradcam(model, x: torch.Tensor, feature: str | None = None) -> torch.Tensor:
    with torch.enable_grad():
        logit, f, _, fp = model(x, return_maps=True)
        target = f if feature == 'f' or not model.use_segm else fp
        gradients = []
        handle = target.register_hook(lambda grad: gradients.append(grad))
        try:
            torch.autograd.grad(logit.sum(), target)
        finally:
            handle.remove()
        weights = gradients[0].mean((2, 3), keepdim=True)
        cam = F.relu((weights * target).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        lo = cam.flatten(1).amin(1)[:, None, None]
        hi = cam.flatten(1).amax(1)[:, None, None]
        return ((cam-lo)/(hi-lo).clamp_min(1e-8)).clamp(0, 1).detach()
