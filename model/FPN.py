import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FPN(nn.Module):

    def __init__(self, in_channels):
        super(FPN, self).__init__()

        self.conv1 = DepthwiseSeparableConv(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1)

        self.toplayer = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.latlayer1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)

        self.smooth1 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3 = DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        p3 = self.toplayer(c3)
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer2(c1))

        p3 = self.smooth3(p3)
        p2 = self.smooth1(p2)
        p1 = self.smooth2(p1)

        return p1, p2, p3


if __name__ == "__main__":
    
    input_tensor = torch.randn(128, 8, 32, 32)
    model = FPN(in_channels=8)
    p1, p2, p3 = model(input_tensor)
    print(f"p1: {p1.size()}")
    print(f"p2: {p2.size()}")
    print(f"p3: {p3.size()}")
