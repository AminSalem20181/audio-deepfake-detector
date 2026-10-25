import torch
from sdeseg.models import Sdeseg
from sdeseg.explain import gradcam

def test_gradcam():
    c = gradcam(Sdeseg().eval(), torch.randn(2,1,64,400)); assert c.shape == (2,64,400); assert float(c.min()) >= 0 and float(c.max()) <= 1
