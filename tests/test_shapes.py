import torch
from sdeseg.models import Sdeseg

def test_shapes():
    y, f, a, fp = Sdeseg()(torch.randn(2,1,64,400), return_maps=True)
    assert y.shape == (2,1) and f.shape == (2,128,8,50) and a.shape == (2,1,8,50) and fp.shape == f.shape
