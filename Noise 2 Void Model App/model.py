"""
N2V용 U-Net 모델. 2D / 3D 두 가지 모드 지원.
"""
import torch
import torch.nn as nn


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    """
    2D U-Net.
    Band별 처리 모드용. in_channels=1, 학습/추론 시 band를 batch 또는 채널로 처리.
    """
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.d1 = DoubleConv2D(in_channels, base)
        self.d2 = DoubleConv2D(base, base * 2)
        self.d3 = DoubleConv2D(base * 2, base * 4)
        self.d4 = DoubleConv2D(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.c3 = DoubleConv2D(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv2D(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv2D(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        u3 = self.u3(d4)
        u3 = self.c3(torch.cat([u3, d3], dim=1))
        u2 = self.u2(u3)
        u2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(u2)
        u1 = self.c1(torch.cat([u1, d1], dim=1))
        return self.out(u1)


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    3D U-Net.
    spatial 2D + spectral 1D 동시 처리.
    입력 shape: (B, 1, D, H, W) where D = spectral chunk
    """
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.d1 = DoubleConv3D(in_channels, base)
        self.d2 = DoubleConv3D(base, base * 2)
        self.d3 = DoubleConv3D(base * 2, base * 4)

        self.pool = nn.MaxPool3d(2)

        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.c2 = DoubleConv3D(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.c1 = DoubleConv3D(base * 2, base)

        self.out = nn.Conv3d(base, out_channels, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))

        u2 = self.u2(d3)
        u2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(u2)
        u1 = self.c1(torch.cat([u1, d1], dim=1))
        return self.out(u1)


def build_model(mode: str = "2D"):
    if mode == "2D":
        return UNet2D()
    elif mode == "3D":
        return UNet3D()
    else:
        raise ValueError(f"Unknown mode: {mode}")
