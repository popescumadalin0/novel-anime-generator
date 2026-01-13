from torch import nn

from hyperparameters import IMAGE_SIZE

class DCGANGenerator(nn.Module):
    def __init__(self,latent_size):
        super(DCGANGenerator,self).__init__()
        """
        Initialize the DCGANGenerator Module
        :param latent_size: The length of the input latent vector
        """
        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=8*IMAGE_SIZE, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(8*IMAGE_SIZE),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8*IMAGE_SIZE, out_channels=4*IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*IMAGE_SIZE),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4*IMAGE_SIZE, out_channels=2*IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*IMAGE_SIZE),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=2*IMAGE_SIZE, out_channels=IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(IMAGE_SIZE),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=IMAGE_SIZE, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 3x64x64 Tensor image as output
        """
        x = self.conv_block1(x)
        return x