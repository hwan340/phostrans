# Maybe rewrite this as a function so the forward pass is clear, the module seems to break back propagation

import numpy as np
# import modules.grid
from itertools import permutations
import cv2
from PIL import Image
import torch
import scipy.io # for loading .mat file
from torchvision import transforms
import torch.nn as nn

class Simulator(nn.Module):
    def __init__(self, grid, device, imgsize = 256):
        super().__init__()
        # self.phoscoding = phoscoding # this is the phosphene coding (level of brightness), output from phosphene_encoder.py
        self.grid = grid * 512/imgsize  # this should be a phosphene map, output from matlab, resize the coordinates to the size of the image (512*512)
        self.device = device
        self.size = 512

    def phos_img(self, phoscoding):
        img = torch.zeros(1, 1, self.size, self.size, device=self.device)
        n = len(phoscoding)
        columnsInImage, rowsInImage = torch.meshgrid(torch.arange(1, self.size + 1, device=self.device), torch.arange(1, self.size + 1, device=self.device))
        x = self.grid[:, 0] # x coordinate of phosphene in pixel
        y = self.grid[:, 1] # y coordinate of phosphene in pixel
        radii = self.grid[:, 2] # radius of phosphene in pixel

        for i in range(n):
            brightness = phoscoding[i]
            phosphene = self.render_phosphene(y[i], x[i], radii[i], rowsInImage, columnsInImage) * brightness
            phosphene = phosphene[None, None]
            # Set parameters for Gaussian blur
            sigma = float(radii[i] / 3)
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            # Apply Gaussian blur
            gau_phosphene = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(phosphene)

            gau_phosphene[gau_phosphene < 0.05] = 0
            img += gau_phosphene

        img[img > 1] = 1
        if torch.max(img) != 0:  # normalization, otherwise the image will be white and show all white
            img = img / torch.max(img)

        return img

    def render_phosphene(self, y, x, radius, rowsInImage, columnsInImage):
        phosphene = torch.square(rowsInImage - x) + torch.square(columnsInImage - y) <= radius ** 2
        return phosphene

    def forward(self, phoscoding):
        return self.phos_img(phoscoding)

# convert everything to pytorch
# Set the device to either 'cuda' for GPU or 'cpu' for CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phosnum = 1200
    phoscoding = torch.ones(phosnum, device=device)
    phos_map_mat = scipy.io.loadmat('data/grids/100610_12ea_UEA_pm_pix.mat')
    grid = phos_map_mat['grid']
    simu = Simulator(grid/1080*512, device, imgsize=512)
    img = simu(phoscoding)

    if img is None:
        print("Check file path")
    else:
        # cv2.imshow("Display", img)
        # cv2.waitKey()
        formatted = (img * 255 / torch.max(img)).byte().numpy().squeeze()
        img = Image.fromarray(formatted)
        img.show()


if __name__ == "__main__":
    test()
