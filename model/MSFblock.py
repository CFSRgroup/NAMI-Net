import torch
import torch.nn as nn

# References & Acknowledgments
# Portions of the code in this file were adapted from the following source:
# https://github.com/xiely-123/shisrcnet

class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MSFblock(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock, self).__init__()
        out_channels = in_channels

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE3 = oneConv(in_channels, in_channels, 1, 0, 1)

    def forward(self, x0, x1, x2):
        y0 = x0
        y1 = x1
        y2 = x2

        y0_weight = self.SE1(self.gap(x0))
        y1_weight = self.SE2(self.gap(x1))
        y2_weight = self.SE3(self.gap(x2))

        weight = torch.cat([y0_weight, y1_weight, y2_weight], 2)
        weight = self.softmax(self.Sigmoid(weight))

        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)

        x_att = y0_weight * y0 + y1_weight * y1 + y2_weight * y2
        return self.project(x_att)


