"""SDE-Seg model and the standard-convolution comparison model."""
from __future__ import annotations

import torch
from torch import nn


class SepBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class Sdeseg(nn.Module):
    """SDE-Seg; forward returns logit and intermediate F/A maps when requested."""
    def __init__(self, use_segm: bool = True) -> None:
        super().__init__()
        self.use_segm = use_segm
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.block1 = SepBlock(16, 32)
        self.block2 = SepBlock(32, 64)
        self.block3 = SepBlock(64, 128)
        self.block4 = SepBlock(128, 128)
        self.pool = nn.MaxPool2d(2)
        self.segm = nn.Conv2d(128, 1, 1, bias=True)
        self.head = nn.Linear(128, 1, bias=True)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool(self.block1(x))
        x = self.pool(self.block2(x))
        x = self.pool(self.block3(x))
        return self.block4(x)

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        f = self.features(x)
        a = torch.sigmoid(self.segm(f)) if self.use_segm else torch.ones_like(f[:, :1])
        fp = f * a
        logit = self.head(fp.mean(dim=(2, 3)))
        if return_maps:
            return logit, f, a, fp
        return logit

    @property
    def last_feature_name(self) -> str:
        return "fp" if self.use_segm else "f"


def standard_conv_twin() -> nn.Module:
    class Block(nn.Module):
        def __init__(self, ci: int, co: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
    class Twin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1,16,3,padding=1,bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
            self.b1, self.b2, self.b3, self.b4 = Block(16,32), Block(32,64), Block(64,128), Block(128,128)
            self.pool = nn.MaxPool2d(2); self.segm = nn.Conv2d(128,1,1); self.head = nn.Linear(128,1)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x=self.stem(x); x=self.pool(self.b1(x)); x=self.pool(self.b2(x)); x=self.pool(self.b3(x)); f=self.b4(x)
            return self.head((f*torch.sigmoid(self.segm(f))).mean((2,3)))
    return Twin()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
