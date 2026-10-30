"""Delete input-resolution Grad-CAM cells, with matched random controls."""
from __future__ import annotations
import torch
from .gradcam import gradcam


def deletion_drop(model, x: torch.Tensor, percentages=(5,10,20,30), random_control=False):
    model.eval()
    with torch.no_grad():
        base = model(x).sigmoid().flatten()
    salience = torch.rand_like(x[:, 0]) if random_control else gradcam(model, x).detach()
    order = salience.flatten(1).argsort(dim=1, descending=True)
    result = {}
    for p in percentages:
        if not 0 < p <= 100:
            raise ValueError('Deletion percentage must be in (0,100]')
        masked = x.clone().flatten(1)
        k = max(1, int(masked.shape[1] * p / 100))
        masked.scatter_(1, order[:, :k], x.mean((1,2,3))[:, None].expand(-1, k))
        with torch.no_grad():
            score = model(masked.view_as(x)).sigmoid().flatten()
        result[p] = float((base-score).mean())
    return result
