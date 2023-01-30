import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import math
from scipy.spatial import distance
from functools import partial
from torchsummary import summary
import numpy as np
from torch.nn.modules.batchnorm import _BatchNorm
nonlinearity = partial(F.relu, inplace=True)


class AANet(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super(AANet, self).__init__()

        resnet = models.resnet34(pretrained=False)
        self.inc = inconv(num_channels, 64)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = down(512,512)

        # self.dblock = DACblock(512)
        # self.spp = SPPblock(512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, num_classes)

        self.pamlock = AAMBlock(512)

    def forward(self, x):
        # Encoder
        x = self.inc(x)        #64*128*128
        e1 = self.encoder1(x)         #64*128*128
        e2 = self.encoder2(e1)        #128*64*64
        e3 = self.encoder3(e2)        #256*32*32
        e4 = self.encoder4(e3)        #512*16*16
        e5 = self.center(e4)          #512*8*8

        e5_2 = self.pamlock(e5)

        d4 = self.up1(e5_2, e4)
        d3 = self.up2(d4, e3)
        d2 = self.up3(d3, e2)
        d1 = self.up4(d2, e1)
        out = self.outc(d1)
        return out


class AAMBlock(nn.Module):
    def __init__(self, in_dim):
        super(AAMBlock, self).__init__()
        self.chanel_in = in_dim
        self.conv_out = nn.Conv2d(in_channels=in_dim*2, out_channels=in_dim, kernel_size=1)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=int(in_dim//8), kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=int(in_dim//8), kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = torch.cat((out,x),1)

        out = self.conv_out(out)

        # out = self.gamma*out + x
        return out


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


