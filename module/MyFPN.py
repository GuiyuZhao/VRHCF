import torch.nn as nn
import torch.nn.functional as F


# The initialization requires a list representing the number of bottlenecks in each stage of RESNET
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.inplanes = 32
        # Build C2, C3, C4, C5 from the bottom up
        self.Conv1=nn.Conv2d(32, 64, 1)
        self.BN1=nn.BatchNorm2d(64)
        self.Conv2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.BN2 = nn.BatchNorm2d(128)
        self.Conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.BN3 = nn.BatchNorm2d(256)

        self.toplayer = nn.Conv2d(256, 64, 1, 1, 0)

        self.smooth1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth3 = nn.Conv2d(64, 64, 3, 1, 1)

        self.latlayer1 = nn.Conv2d(128, 64, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(64, 64, 1, 1, 0)


    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):

        c2 = self.Conv1(x)
        c2 = self.BN1(c2)
        c3 = self.Conv2(c2)
        c3 = self.BN2(c3)
        c4 = self.Conv3(c3)
        c4 = self.BN3(c4)

        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2= self._upsample_add(p3, self.latlayer2(c2))

        p2 = self.smooth3(p2)
        return p2
