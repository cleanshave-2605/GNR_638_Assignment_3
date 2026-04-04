import torch
import torch.nn as nn
from typing import List
# from blocks import *
from utils import *

# Stage builder
class bottleneck_layer(nn.Module):
    expansion = 4

    def __init__(
            self, 
            inplanes, 
            planes, 
            stride=1, 
            downsample=None,
            start_block=False, 
            end_block=False, 
            exclude_bn0=False
        ):

        super(bottleneck_layer, self).__init__()
        
        if not start_block and not exclude_bn0:
            self.bn0 = nn.BatchNorm2d(inplanes)
        
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = conv1x1(planes, planes * self.expansion)
        if start_block or end_block:
            self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.start_block, self.end_block, self.exclude_bn0 = start_block, end_block, exclude_bn0


    def forward(self, x):
        identity = x
        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out)
        
        if self.start_block:
            out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        
        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)
        return out
    


class iresnet(nn.Module):
    def __init__(self, layers, num_classes=10):
        super(iresnet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)


    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        
        if ((stride != 1) or (self.inplanes != planes * 4)):
            shortcut = []
            
            if stride != 1:
                shortcut.append(nn.AvgPool2d(2, stride=2, ceil_mode=True))
            
            shortcut.append(conv1x1(self.inplanes, planes * 4))
            shortcut.append(nn.BatchNorm2d(planes * 4))
            downsample = nn.Sequential(*shortcut)

        layers = [bottleneck_layer(self.inplanes, planes, stride, downsample, start_block=True)]
        self.inplanes = planes * 4
        for i in range(1, blocks - 1):
            layers.append(bottleneck_layer(self.inplanes, planes, exclude_bn0=(i==1)))
        
        layers.append(bottleneck_layer(self.inplanes, planes, end_block=True, exclude_bn0=(blocks==2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))