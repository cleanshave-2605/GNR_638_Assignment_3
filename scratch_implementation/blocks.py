import torch.nn as nn
from utils import *

# Improved Projection Shortcut
class ImprovedProjectionShortcut(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.conv = conv1x1(in_planes, out_planes)
        self.bn   = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(self.pool(x)))


# CIFAR Bottleneck Blocks
class StartBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes,
                 stride = 1,
                 use_improved_shortcut = True):
        super().__init__()
        out_planes = planes * self.expansion

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(planes, out_planes)
        self.bn3   = nn.BatchNorm2d(out_planes)

        # Shortcut: only needed when spatial size or channels change
        if stride != 1 or in_planes != out_planes:
            if use_improved_shortcut and stride > 1:
                self.shortcut = ImprovedProjectionShortcut(
                    in_planes, out_planes, stride)
            else:
                # Channel-only projection (no spatial change)
                self.shortcut = nn.Sequential(
                    conv1x1(in_planes, out_planes),
                    nn.BatchNorm2d(out_planes)
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        return out + identity 


class MiddleBlock(nn.Module):
    expansion = 4

    def __init__(self, planes, first_in_stage = False):
        super().__init__()
        self.first_in_stage = first_in_stage
        ch = planes * self.expansion

        if not first_in_stage:
            self.bn0 = nn.BatchNorm2d(ch)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(ch, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(planes, ch)

    def forward(self, x):
        identity = x

        if not self.first_in_stage:
            out = self.relu0(self.bn0(x)) 
        else:
            out = self.relu0(x)

        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        return out + identity


class EndBlock(nn.Module):
    expansion = 4

    def __init__(self, planes):
        super().__init__()
        ch = planes * self.expansion

        self.bn0   = nn.BatchNorm2d(ch)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(ch, planes)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(planes, ch)

        # Applied to the full post-addition signal (key difference from Middle)
        self.bn_out   = nn.BatchNorm2d(ch)
        self.relu_out = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x

        out = self.relu0(self.bn0(x))
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.conv3(out)

        return self.relu_out(self.bn_out(out + identity))