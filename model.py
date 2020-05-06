import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, channels, upscale_factor, concat=False):
        super(UpsampleBlock, self).__init__()
        if concat:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv = nn.Conv2d(channels, channels * upscale_factor, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_bn_relu(  3,  32, kernel_size=3) # 32x256x256
        self.enc2 = self.conv_bn_relu( 32,  64, kernel_size=3, pool_kernel=2) # 64x128x128
        self.enc3 = self.conv_bn_relu( 64 + 64,  128, kernel_size=3, pool_kernel=2) # 128x64x64
        self.enc4 = self.conv_bn_relu(128, 256, kernel_size=3, pool_kernel=2) # 256x32x32
        self.enc5 = self.conv_bn_relu(256, 512, kernel_size=3, pool_kernel=2) # 512x16x16

        self.sub1 = self.conv_bn_relu(  3,  32, kernel_size=3) # 32x256x256
        self.sub2 = self.conv_bn_relu( 32,  64, kernel_size=3, pool_kernel=2) # 64x128x128

        self.dec1 = UpsampleBlock(512, 2) # 256x32x32
        self.dec2 = UpsampleBlock(256 + 256, 2, True) # 128x64x64
        self.dec3 = UpsampleBlock(128 + 128, 2, True) # 64x128x128
        self.dec4 = UpsampleBlock( 64 +  64, 2, True) # 32x256x256
        self.dec5 = nn.Sequential(
            nn.Conv2d(32 + 32, 3, kernel_size=5, padding=2),
            nn.Tanh()
        )

        self.res1 = ResBlock(256, 256)
        self.res2 = ResBlock(128, 128)
        self.res3 = ResBlock( 64,  64)

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None):
        layers = []
        if pool_kernel is not None:
            layers.append(nn.AvgPool2d(pool_kernel))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, h): # x:input, h:hint
        x1 = self.enc1(x)
        x2 = self.enc2(x1)

        h = self.sub1(h)
        h = self.sub2(h)

        x3 = self.enc3(torch.cat([x2, h], dim=1))
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        out = self.dec1(x5)
        out = self.res1(out)
        out = self.dec2(torch.cat([out, x4], dim=1))
        out = self.res2(out)
        out = self.dec3(torch.cat([out, x3], dim=1))
        out = self.res3(out)
        out = self.dec4(torch.cat([out, x2], dim=1))
        out = self.dec5(torch.cat([out, x1], dim=1))
        return out

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.conv1 = self.conv_bn_relu(  3,  16, kernel_size=5, reps=1) # 16x256x256
        self.conv2 = self.conv_bn_relu( 16,  32, pool_kernel=2) # 32x128x128
        self.conv3 = self.conv_bn_relu( 32,  64, pool_kernel=2) # 64x64x64
        self.conv4 = self.conv_bn_relu( 64, 128, pool_kernel=2) # 128x32x32
        self.conv5 = self.conv_bn_relu(128, 256, pool_kernel=2) # 256x16x16
        self.patch = nn.Conv2d(256, 1, kernel_size=1) # 1x16x16

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, pool_kernel=None, reps=2):
        layers = []
        for i in range(reps):
            if i == 0 and pool_kernel is not None:
                layers.append(nn.AvgPool2d(pool_kernel))
            layers.append(nn.utils.spectral_norm(nn.Conv2d(in_ch if i == 0 else out_ch,
                                    out_ch, kernel_size, padding=(kernel_size - 1) // 2)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        out = self.patch(out)
        return out

