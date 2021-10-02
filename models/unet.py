import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            m.apply(weights_init_xavier)

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            m.apply(weights_init_xavier)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetDecoder(nn.Module):
    def __init__(self, out_channel = 1):
        super(UnetDecoder, self).__init__()
        self.filters = [32, 64, 128, 256, 512]
        self.is_deconv = True
        self.up_concat4 = unetUp(self.filters[4], self.filters[3], self.is_deconv)
        self.up_concat3 = unetUp(self.filters[3], self.filters[2], self.is_deconv)
        self.up_concat2 = unetUp(self.filters[2], self.filters[1], self.is_deconv)
        self.up_concat1 = unetUp(self.filters[1], self.filters[0], self.is_deconv)
        self.upsample = nn.Sequential(
            nn.Conv2d(self.filters[0],self.filters[0],3,1,1),
            nn.Upsample(None, 2, 'nearest')
        )
        # self.relu = nn.ReLU()
        self.final = nn.Conv2d(self.filters[0], out_channel, 1)# 1x1的卷积，只改变通道数，不改变其他信息

    def forward(self, conv):
        up4 = self.up_concat4(conv[3], conv[4])
        up3 = self.up_concat3(conv[2], up4)
        up2 = self.up_concat2(conv[1], up3)
        up1 = self.up_concat1(conv[0], up2)
        up0 = self.upsample(up1)
        return self.final(up0)

# 这个结构是没有做放大的操作的
class Backend(nn.Module):
    def __init__(self, c1, c2):
        super(Backend, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1, c1, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1, c1, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1, c1//2, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1//2, c1//4, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1//4, c1//8, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1//8, c2, 1, 1),
        )

    def forward(self, x):
        return self.model(x)

class BackendUpSample(nn.Module):
    def __init__(self, c1, c2):
        super(BackendUpSample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1, c1, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1, c1, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1, c1//2, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1//2, c1//4, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1//4, c1//8, 3, 1, 2, 2),
            nn.ReLU(),
            nn.Conv2d(c1//8, c2, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    decoder = UnetDecoder()
    print(decoder.parameters())
    paramCount = 0
    for name, param in decoder.named_modules():
        print('name => ', name)






