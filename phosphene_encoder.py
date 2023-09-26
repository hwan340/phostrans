#This is the encoder model for the CycleGAN model, which encode value for each phosphene
import torch
import torch.nn as nn
from torchinfo import summary

# Define the block for generator model
class ConvBlock(nn.Module):
    # Initalize the class with in_channels, out_channels, down sampling, use of activation and key word arguments
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # define the conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) # key word would be kernal size, stride and padding
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels), # we replace instance norm with batch norm
            nn.LeakyReLU(inplace=True) if use_act else nn.Identity()
        )

    # forward method
    def forward(self, x):
        return self.conv(x)

class ConvPoolBlock(nn.Module):
    # Initalize the class with in_channels, out_channels, down sampling, use of activation and key word arguments
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        # define the conv block
        self.convpool = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs), # key word would be kernal size, stride and padding
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True) if use_act else nn.Identity()
        )

    # forward method
    def forward(self, x):
        return self.convpool(x)

# define the residual block
class ResidualBlock(nn.Module):
    # initialize the class with channels
    def __init__(self, channels):
        super().__init__()
        # define the residual block
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )

    # forward method
    def forward(self, x):
        return x + self.block(x)

# define the generator model
class Phoscoder(nn.Module):
    # initialize the class with img_channels, num_features, num_residuals
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        # initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            # nn.InstanceNorm2d(num_features),
            nn.LeakyReLU(inplace=True)
        )
        # down sampling
        self.down_blocks = nn.ModuleList(
            [
                ConvPoolBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvPoolBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
                ConvPoolBlock(num_features * 4, num_features * 8, kernel_size=3, stride=2, padding=1),
                # ConvBlock(num_features * 4, num_features * 8, kernel_size=3, stride=2, padding=1),
                # ConvBlock(num_features * 8, num_features * 16, kernel_size=3, stride=2, padding=1),
                # ConvBlock(num_features * 16, num_features * 32, kernel_size=3, stride=2, padding=1),
            ]
        )
        # residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*8) for _ in range(num_residuals)]
        )
        # up sampling
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*8, num_features*4, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        # final layer, change this so the output is a vector representing phosphene image
        self.last = nn.Linear(num_features*2*32*32, 1200) # fully connected layer to get 1200 phosphene encoding

    # forward method
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)

            # print(x.size())
        x = self.res_blocks(x)
        # print(x.size())
        for layer in self.up_blocks:
            x = layer(x)
        x = torch.flatten(x)
        return self.last(x) # was tanh

# test the generator model
def test():
    img_channels = 1
    img_size = 512
    x = torch.randn((1, img_channels, img_size, img_size))
    gen = Phoscoder(img_channels=1, num_features = 16, num_residuals=4)
    gen(x)
    print(gen(x).shape)
    summary(gen,(1, 1, 512, 512))

if __name__ == "__main__":
    test()


# comment is very important when use copilot
# copilot works well with machine learning/deep learning project which has a large community
# still needs minor modification even for existing model
# copilot take your previous code into consideration

# Potential issue is copilot could suggest someone else's code and you are not aware of it
# In pycharm you cannot select partial suggestion

