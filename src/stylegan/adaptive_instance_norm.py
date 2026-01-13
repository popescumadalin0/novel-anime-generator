from torch import nn


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, num_features, w_dim):
        super(AdaptiveInstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

        # Style modulation parameters
        self.style_scale = nn.Linear(w_dim, num_features)
        self.style_bias = nn.Linear(w_dim, num_features)

    def forward(self, x, w):
        # Normalize the input
        x = self.norm(x)

        # Apply style modulation
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)

        return style_scale * x + style_bias
