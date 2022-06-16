import torch.nn.functional as F
import torchvision.models as models
from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
        )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_planes // 4, in_planes // 4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.tp_conv(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = x2 + self.conv2(x1)

        return x


class LinkNet(nn.Module):
    def __init__(self, num_classes=21):
        super(LinkNet, self).__init__()

        base = models.resnet18()

        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 2, 2, 0)

    def forward(self, x):
        x = self.in_block(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, x)

        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y
