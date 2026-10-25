import torch

def spec_augment(x: torch.Tensor, time_max: int=40, freq_max: int=8, masks: int=2) -> torch.Tensor:
    out=x.clone()
    for _ in range(masks):
        t=int(torch.randint(0,time_max+1,()).item()); f=int(torch.randint(0,freq_max+1,()).item())
        if t:
            start=int(torch.randint(0,max(1,out.shape[-1]-t+1),()).item()); out[..., :, start:start+t] = 0
        if f:
            start=int(torch.randint(0,max(1,out.shape[-2]-f+1),()).item()); out[..., start:start+f, :] = 0
    return out
