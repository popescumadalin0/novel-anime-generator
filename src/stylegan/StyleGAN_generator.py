import torch
from torch import nn

from mapping_network import MappingNetwork
from style_block import StyleBlock


class StyleGANGenerator(nn.Module):
    """
    StyleGAN Generator with mapping network and synthesis network
    Generates 64x64 anime faces
    """

    def __init__(self, latent_dim=512, w_dim=512, img_channels=3):
        super(StyleGANGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.w_dim = w_dim

        
        self.mapping = MappingNetwork(latent_dim, w_dim)

        
        self.constant_input = nn.Parameter(torch.randn(1, 512, 4, 4))

        
        self.style_blocks = nn.ModuleList([
            StyleBlock(512, 512, w_dim, upsample=False),  
            StyleBlock(512, 512, w_dim, upsample=True),  
            StyleBlock(512, 256, w_dim, upsample=True),  
            StyleBlock(256, 128, w_dim, upsample=True),  
            StyleBlock(128, 64, w_dim, upsample=True),  
        ])

        
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(512, img_channels, 1),
            nn.Conv2d(512, img_channels, 1),
            nn.Conv2d(256, img_channels, 1),
            nn.Conv2d(128, img_channels, 1),
            nn.Conv2d(64, img_channels, 1),
        ])

    def forward(self, z, return_w=False):
        batch_size = z.shape[0]

        
        w = self.mapping(z)

        
        x = self.constant_input.repeat(batch_size, 1, 1, 1)

        
        for i, style_block in enumerate(self.style_blocks):
            x = style_block(x, w)

        
        img = self.to_rgb[-1](x)
        img = torch.tanh(img)

        if return_w:
            return img, w
        return img

    def style_mixing(self, z1, z2, mix_layer=2):
        """
        Style mixing: use different w vectors at different layers
        """
        batch_size = z1.shape[0]

        w1 = self.mapping(z1)
        w2 = self.mapping(z2)

        x = self.constant_input.repeat(batch_size, 1, 1, 1)

        for i, style_block in enumerate(self.style_blocks):
            w = w1 if i < mix_layer else w2
            x = style_block(x, w)

        img = self.to_rgb[-1](x)
        img = torch.tanh(img)

        return img