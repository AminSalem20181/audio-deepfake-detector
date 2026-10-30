"""Class-balanced clean BCE and differentiable two-pass SEGM stability."""
from __future__ import annotations
import torch
from torch import nn


def sde_loss(model, x: torch.Tensor, y: torch.Tensor, pos_weight: float,
             lam: float = 0.1, sigma: float = 0.1):
    if pos_weight <= 0 or lam < 0 or sigma < 0:
        raise ValueError('pos_weight must be positive; lambda and sigma nonnegative')
    clean_logit, _, attention, _ = model(x, return_maps=True)
    _, _, noisy_attention, _ = model(x + torch.randn_like(x) * sigma, return_maps=True)
    bce = nn.functional.binary_cross_entropy_with_logits(
        clean_logit.squeeze(1), y.float(), pos_weight=x.new_tensor(pos_weight))
    stability = (attention-noisy_attention).abs().mean()
    return bce + lam*stability, bce.detach(), stability.detach()
