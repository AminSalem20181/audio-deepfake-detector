from __future__ import annotations
import torch

def deletion_drop(model, x: torch.Tensor, percentages=(5,10,20,30), random_control=False):
    model.eval()
    with torch.no_grad(): base=torch.sigmoid(model(x)).squeeze(1)
    result={}
    for p in percentages:
        cam = torch.rand_like(x) if random_control else model(x, return_maps=True)[2].detach()
        flat=cam.flatten(1); k=max(1, int(flat.shape[1] * p / 100)); idx=flat.topk(k, dim=1).indices
        masked=x.clone().flatten(1); mean=x.mean(dim=(1,2,3), keepdim=True).flatten(1)
        masked.scatter_(1, idx, mean.expand(-1, masked.shape[1]).gather(1, idx)); masked=masked.view_as(x)
        with torch.no_grad(): score=torch.sigmoid(model(masked)).squeeze(1)
        result[p]=float((base-score).mean())
    return result
