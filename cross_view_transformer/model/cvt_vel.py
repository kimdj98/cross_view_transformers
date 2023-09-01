import torch
import torch.nn as nn

class VelocityPrediction(nn.Module):
    def __init__(
        self,
        cvt,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()
        self.cvt = cvt

        self.UNet = UNet(4, 2)


    def forward(self, batch):
        curr = self.cvt(batch)

        batch['image'] = batch['prev_image']
        prev = self.cvt(batch)

        x = torch.concat((curr['bev'].detach(), curr['center'].detach(), prev['bev'].detach(), prev['center'].detach()), dim=1)
        x = self.UNet(x)

        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(256, 128)  # +128 for concatenation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(128 + 128, 64)  # +64 for concatenation
        )
        self.outc = nn.Conv2d(64 + 64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = self.up2(torch.cat([x2, x], dim=1))
        logits = self.outc(torch.cat([x1, x], dim=1))
        return logits