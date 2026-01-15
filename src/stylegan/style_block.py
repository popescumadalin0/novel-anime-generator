from torch import nn

from adaptive_instance_norm import AdaptiveInstanceNorm
from noise_injection import NoiseInjection


class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, upsample=True):
        super(StyleBlock, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise = NoiseInjection()
        self.adain = AdaptiveInstanceNorm(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv(x)
        x = self.noise(x, noise)
        x = self.adain(x, w)
        x = self.activation(x)

        return x