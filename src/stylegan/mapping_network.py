from torch import nn


class MappingNetwork(nn.Module):

    def __init__(self, latent_dim=512, hidden_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(latent_dim if i == 0 else hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2)
            ])

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)
