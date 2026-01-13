from torch import nn

from hyperparameters import IMAGE_SIZE

class DCGANDiscriminator(nn.Module):
    def __init__(self,inchannels):
        super(DCGANDiscriminator,self).__init__()
        """
        Initialize the DCGANDiscriminator Module
        :param inchannels: The depth of the first convolutional layer
        """
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=IMAGE_SIZE, out_channels=2*IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=2*IMAGE_SIZE, out_channels=4*IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=4*IMAGE_SIZE, out_channels=8*IMAGE_SIZE, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8*IMAGE_SIZE),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels=8*IMAGE_SIZE, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self,x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: Discriminator logits; the output of the neural network
        """
        x = self.conv_block1(x)
        return x