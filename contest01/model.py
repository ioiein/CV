import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
from utils import NUM_PTS, CROP_SIZE


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class SEModule(nn.Module):
    '''Squeeze and Excitation Module'''
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x


class RESNEXT_st(nn.Module):
    def __init__(self):
        super(RESNEXT_st, self).__init__()
        model = models.resnext50_32x4d()
        model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
        #checkpoint = torch.load("./runs/baseline_full4_best.pth", map_location='cpu')
        #model.load_state_dict(checkpoint, strict=True)
        self.base_net = nn.Sequential(*list(model.children())[:-2])
        out_size = model.fc.in_features
        self.linear7 = ConvBlock(out_size, out_size, (4, 4), 1, 0, dw=True, linear=True) #(7x7)
        self.linear1 = ConvBlock(out_size, out_size, 1, 1, 0, linear=True)
        self.attention = SEModule(out_size, 8)
        self.fc = nn.Linear(out_size, 2 * NUM_PTS, bias=True)


    def forward(self, x):
        x = self.base_net(x)
        x = self.attention(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AvgResNet(nn.Module):
    def __init__(self, output_size=2*NUM_PTS):
        super(AvgResNet, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.requires_grad_(True)

        fc_input = 512 * (CROP_SIZE // 32) ** 2
        s1_input = 256 * 4 ** 2
        s2_input = 128 * 8 ** 2
        self.model.fc = nn.Linear(fc_input + s1_input + s2_input, output_size, bias=True)
        self.model.fc.requires_grad_(True)
        self.model.layer4.requires_grad_(True)
        self.model.layer3.requires_grad_(True)
        self.model.layer2.requires_grad_(True)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((4, 4))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        x4 = torch.flatten(x4, 1)
        x3 = self.avg_pool1(x3)
        x3 = torch.flatten(x3, 1)
        x2 = self.avg_pool2(x2)
        x2 = torch.flatten(x2, 1)

        out = torch.cat([x4, x3, x2], 1)
        out = self.model.fc(out)
        return out

