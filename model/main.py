import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from DMRE import DMRE
from FPN import FPN
from SWDA import DilateBlock
from msda import CTIModule
from MSFblock import MSFblock

torch.autograd.set_detect_anomaly(True)

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.fpn = FPN(in_channels=8)
        self.msda0 = DilateBlock(attn_drop=0.4, proj_drop=0.4, dim=16)
        self.scconv = DMRE(16)
        self.cti = CTIModule(embed_dim=16, num_heads=4, drop_path=0.4)
        self.msf = MSFblock(in_channels=16)

        self.td25616 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=1)


        self.trial_head0 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),

            nn.Linear(8*8*8, 128),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),

        )

        self.trial_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        temporal = x[:, :16, :, :]
        frequency = x[:, 16:, :, :]

        timeout = self.scconv(temporal)
        timeout = F.avg_pool2d(timeout, kernel_size=2, stride=2)
        p1, p2, p3 = self.fpn(frequency)

        p1 = self.td25616(p1)
        p1 = self.msda0(p1)

        p2 = self.td25616(p2)
        p2 = self.msda0(p2)

        p3 = self.td25616(p3)
        p3 = self.msda0(p3)

        o3, o4, o5 = self.cti(timeout, p1, p2, p3)
        out = self.msf(o3, o4, o5)

        outmiddle = self.trial_head0(out)
        out = self.trial_head(outmiddle)

        return outmiddle, out

def EEGNET():
    return EEGNet()


if __name__ == '__main__':
    x = np.random.rand(128, 24, 32, 32)
    x = torch.tensor(x, dtype=torch.float32)
    model = EEGNET()
    outputmiddle, output = model(x)
    print(outputmiddle.shape)
    print(output.shape)




