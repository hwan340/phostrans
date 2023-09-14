#This is the discriminator model for the CycleGAN model
import torch
import torch.nn as nn

# Define the block for discriminator model
class Block(nn.Module):
    # initialize the class with in_channels, out_channels, and stride

    # def __init__(self, in_channels, out_channels, stride):
    #     super(Block, self).__init__()
    #     # convolutional layer
    #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
    #     # batch normalization
    #     self.batch_norm = nn.BatchNorm2d(out_channels)
    #     # leaky relu
    #     self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        # sequential convolutional layer
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,4, stride,1, padding_mode="reflect"),
                                    nn.InstanceNorm2d(out_channels), # instance norm
                                    nn.LeakyReLU(0.2)) # leaky relu

    # forward method
    def forward(self, x):
        # convolutional layer
        x = self.conv(x)
        return x


# All following code is generated by copilot
# define the discriminator model
class Discriminator(nn.Module):
    # initialize the class with in_channels, features
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2) # inplace=True
        )
        # define the discriminator blocks
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        # add the layers to the model
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    # forward method
    # def forward(self, x):
    #     # initial convolutional layer
    #     x = self.initial(x)
    #     # discriminator blocks
    #     x = self.model(x)
    #     return x

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

# test the model
def test():
    x = torch.randn((1, 1, 512, 512))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()