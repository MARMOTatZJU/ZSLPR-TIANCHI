import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from .se_module import SELayer


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes ,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes ),
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            BasicBlock(64,128,2),
            BasicBlock(128, 128, 1),
            BasicBlock(128, 128, 1),

        )
        self.stage3 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
        )
        self.stage4 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512, 1),
            BasicBlock(512, 512, 1),
            BasicBlock(512, 512, 1),

        )
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,  reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.stride = stride
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.stride != 1:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class se_resnet(nn.Module):

    def __init__(self, num_classes=1000):
        super(se_resnet, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            SEBasicBlock(64,128,2),
            SEBasicBlock(128, 128, 1),
            SEBasicBlock(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            SEBasicBlock(128, 256, 2),
            SEBasicBlock(256, 256, 1),
            SEBasicBlock(256, 256, 1),
            SEBasicBlock(256, 256, 1),
        )
        self.stage4 = nn.Sequential(
            SEBasicBlock(256, 512, 2),
            SEBasicBlock(512, 512, 1),
            SEBasicBlock(512, 512, 1),

        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x, VisFeat=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if VisFeat:
            return x
        else:
            return self.fc(x)

class se_resnet_smaller(nn.Module):

    def __init__(self, num_classes=1000):
        super(se_resnet_smaller, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            SEBasicBlock(64,128,2),
            SEBasicBlock(128, 128, 1),
            SEBasicBlock(128, 128, 1),
        )
        self.stage3 = nn.Sequential(
            SEBasicBlock(128, 256, 2),
            SEBasicBlock(256, 256, 1),
            SEBasicBlock(256, 256, 1),
        )

        self.stage4 = nn.Sequential(
            SEBasicBlock(256, 512, 2),
            SEBasicBlock(512, 512, 1),
        )

        self.avgpool = nn.AvgPool2d(4)

        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ =='__main__':
    model = se_resnet(300).cuda()
    img = torch.rand(4,3,80,80).cuda()   #416 320     800 608
    out =  model(img)
    att = torch.rand(300,164).cuda()
    res = torch.mm(out,att)
    print(out.size())
    print(res.size())
