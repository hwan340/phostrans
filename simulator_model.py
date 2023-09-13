import numpy as np
# import modules.grid
from itertools import permutations
import cv2
from PIL import Image
import torch
import scipy.io # for loading .mat file


class Simulator:

    def __init__(self, phoscoding, grid, imgsize = 256):
        self.phoscoding = phoscoding # this is the phosphene coding (level of brightness), output from phosphene_encoder.py
        self.grid = grid * 512/imgsize  # this should be a phosphene map, output from matlab, resize the coordinates to the size of the image (512*512)
        self.size = 512
        self.image = self.phos_img()


    def phos_img(self):
        img = np.zeros((self.size, self.size))
        n = len(self.phoscoding)
        columnsInImage, rowsInImage = np.meshgrid(np.arange(1, self.size + 1), np.arange(1, self.size + 1))
        x = self.grid[:, 0] # x coordinate of phosphene in pixel
        y = self.grid[:, 1] # y coordinate of phosphene in pixel
        radii = self.grid[:, 2] # radius of phosphene in pixel

        for i in range(n):
            if self.phoscoding[i] > 0.5:
                phosphene = self.render_phosphene(x[i], y[i], radii[i], rowsInImage, columnsInImage) * 1
                gau_phosphene = cv2.GaussianBlur(np.float32(phosphene), ksize=(0, 0), sigmaX=radii[i] / 3, borderType=cv2.BORDER_REPLICATE)
                gau_phosphene[gau_phosphene < 0.05] = 0
                img = img + gau_phosphene

        img[img > 1] = 1
        if np.max(img) != 0: # normalization, otherwise imange will be white and show all white
            img = img / np.max(img)

        return img

    def render_phosphene(self, x, y, radius, rowsInImage, columnsInImage):
        phosphene = (rowsInImage - y) ** 2 + (columnsInImage - x) ** 2 <= radius ** 2
        return phosphene


def test():
    phosnum = 1200
    phoscoding = np.random.rand(phosnum)
    # phoscoding = np.random.randint(2, size=phosnum)
    # grid = np.random.rand(phosnum, 3)    # x, y, radius
    # grid[:, 0] *=100
    # grid[:, 0] = grid[:, 0] + 64
    # grid[:, 1] *=100
    # grid[:, 1] = grid[:, 1] + 64
    # grid[:, 2] *=3
    phos_map_mat = scipy.io.loadmat('data/grids/100610_12ea_UEA_pm_pix.mat')
    grid = phos_map_mat['grid']
    simu = Simulator(phoscoding, grid/1080*512, imgsize=512)
    img = simu.image

    if img is None:
        print("Check file path")
    else:
        # cv2.imshow("Display", img)
        # cv2.waitKey()
        formatted = (img * 255 / np.max(img)).astype('uint8')
        img = Image.fromarray(formatted)
        img.show()


if __name__ == "__main__":
    test()
