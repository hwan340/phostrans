#This is the generator model for the CycleGAN model
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
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity() # inplace=True copliot is learning form your script
        )

    # forward method
    def forward(self, x):
        return self.conv(x)

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
class Generator(nn.Module):
    # initialize the class with img_channels, num_features, num_residuals
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super().__init__()
        # initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            # nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        # down sampling
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1)
            ]
        )
        # residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        # up sampling
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        # final layer
        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    # forward method
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

# test the generator model
def test():
    img_channels = 1
    img_size = 1080
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)
    summary(gen, (1, 1080, 1080))

if __name__ == "__main__":
    test()


# comment is very important when use copilot
# copilot works well with machine learning/deep learning project which has a large community
# still needs minor modification even for existing model
# copilot take your previous code into consideration

# Potential issue is copilot could suggest someone else's code and you are not aware of it
# In pycharm you cannot select partial suggestion

