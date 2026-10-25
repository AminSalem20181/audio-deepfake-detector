import torch
from torch import nn

def sde_loss(model, x, y, pos_weight: float, lam: float = 0.1, sigma: float = 0.1):
    clean_logit, _, a, _ = model(x, return_maps=True)
    noisy = x + torch.randn_like(x) * sigma
    _, _, a_noisy, _ = model(noisy, return_maps=True)
    bce = nn.functional.binary_cross_entropy_with_logits(clean_logit.squeeze(1), y.float(), pos_weight=torch.tensor(pos_weight, device=x.device))
    return bce + lam * (a - a_noisy).abs().mean(), bce.detach(), (a-a_noisy).abs().mean().detach()
