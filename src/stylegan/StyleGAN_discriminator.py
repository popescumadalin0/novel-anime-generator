from torch import nn


class StyleGANDiscriminator(nn.Module):
    """
    Progressive discriminator for StyleGAN
    """

    def __init__(self, img_channels=3):
        super(StyleGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, img):
        return self.model(img).view(-1)