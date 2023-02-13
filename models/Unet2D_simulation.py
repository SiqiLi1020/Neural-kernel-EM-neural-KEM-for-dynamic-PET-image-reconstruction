import torch
import torch.nn as nn
import torch.nn.functional as F

## The network is based on the paper, titled "K Gong et al, PET Image Reconstruction Using Deep Image Prior"
# 2D Unet; more parameters: 32-64-128.

class double_conv(nn.Module):
    """convolution => [BN] => ReLU) * 2, no downsample"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        )

    def forward(self, x):
        return self.double_conv(x)

class Convout(nn.Module):
    """out image using convolution and relu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )
    def forward(self, x):
        return self.double_conv1(x)

class down_conv(nn.Module):
    """stride convolution => [BN] => ReLU, downsample"""

    def __init__(self, in_channels, out_channels, max=True):
        super().__init__()
        if max:
            self.down_conv = nn.MaxPool2d(2, stride=2)
        else:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.01, inplace=False)
            )
    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling using bilinear or deconv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        else:
            #self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x1, x2):
        x1 = self.relu(self.bn(self.conv(self.up(x1))))
        x1 += x2
        return x1


class UNet(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, max = False, bilinear = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.max = max
# encoder
        self.inc1 = double_conv(in_channels, inter_channels)
        self.down1 = down_conv(inter_channels, inter_channels, max)
        self.inc2 = double_conv(inter_channels,inter_channels*2)
        self.down2 = down_conv(inter_channels*2, inter_channels*2, max)
        self.inc3 = double_conv(inter_channels*2, inter_channels*4)
        self.down3 = down_conv(inter_channels*4, inter_channels*4, max)
        self.inc4 = double_conv(inter_channels*4, inter_channels*8)
# decoder
        self.up1 = Up( inter_channels*8,  inter_channels*4, bilinear)
        self.inc5 = double_conv(inter_channels*4, inter_channels*4)
        self.up2 = Up(inter_channels*4, inter_channels*2, bilinear)
        self.inc6 = double_conv(inter_channels*2, inter_channels*2)
        self.up3 = Up(inter_channels*2, inter_channels, bilinear)
        self.inc7 = double_conv(inter_channels, inter_channels)
        self.out = Convout(inter_channels, out_channels)

    def forward(self, x):
        x1 = self.inc1(x)
        x1_down = self.down1(x1)
        x2 = self.inc2(x1_down)
        x2_down = self.down2(x2)
        x3 = self.inc3(x2_down)
        x3_down = self.down3(x3)
        x4 = self.inc4(x3_down)

        x5 = self.up1(x4,x3)
        x5 = self.inc5(x5)
        x6 = self.up2(x5,x2)
        x6 = self.inc6(x6)
        x7 = self.up3(x6,x1)
        x7 = self.inc7(x7)
        out = self.out(x7)
        return out

if __name__ == '__main__':
    net = UNet(in_channels=1, out_channels=1)
    print(net)




