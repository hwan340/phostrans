import torch
from PIL import Image
import scipy.io
from torchvision import transforms
import numpy as np
# check if everything is differentiable.

class PhosImageFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, phoscoding, grid, imgsize, device):
        """
        :param phoscoding: a 1*1200 tensor, each element is the brightness of the corresponding phosphene
        :param grid: a 1200*3 tensor, each row is the x, y, and radius of a phosphene
        """
        size = 512
        ctx.device = device
        ctx.size = size

        img = torch.zeros(1, 1, size, size, device=device)
        n = len(phoscoding)
        columnsInImage, rowsInImage = torch.meshgrid(torch.arange(1, size + 1, device=device), torch.arange(1, size + 1, device=device))
        x = grid[:, 0]  # x coordinate of phosphene in pixel
        y = grid[:, 1]  # y coordinate of phosphene in pixel
        radii = grid[:, 2]  # radius of phosphene in pixel

        for i in range(n):
            brightness = phoscoding[i]
            phosphene = PhosImageFunction.render_phosphene(y[i], x[i], radii[i], rowsInImage, columnsInImage) * brightness
            phosphene = phosphene[None, None]
            # Set parameters for Gaussian blur
            sigma = float(radii[i] / 3)
            kernel_size = int(2 * np.ceil(2 * sigma) + 1)
            # Apply Gaussian blur
            gau_phosphene = transforms.GaussianBlur(kernel_size= kernel_size, sigma=sigma)(phosphene)

            gau_phosphene[gau_phosphene < 0.05] = 0
            img += gau_phosphene

        img[img > 1] = 1
        if torch.max(img) != 0:  # normalization, otherwise the image will be white and show all white
            img = img / torch.max(img)

        ctx.save_for_backward(phoscoding, img)

        return img

    # @staticmethod
    # def backward(ctx, grad_output):
    #     phoscoding, img = ctx.saved_tensors
    #
    #     # device = ctx.device
    #
    #     # grad_phoscoding = torch.zeros_like(phoscoding)
    #     return grad_output*img, None, None, None

    @staticmethod
    def render_phosphene(y, x, radius, rowsInImage, columnsInImage):
        distance_squared = torch.square(rowsInImage - x) + torch.square(columnsInImage - y)
        phosphene = distance_squared <= radius ** 2
        return phosphene

class Simulator:

    def __init__(self, imgsize=256):
        self.imgsize = imgsize

    def __call__(self, phoscoding, grid, device):
        return PhosImageFunction.apply(phoscoding, grid, self.imgsize, device)

# Example usage:
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phosnum = 1200
    phoscoding = torch.ones(phosnum, device=device)

    phos_map_mat = torch.tensor(scipy.io.loadmat('data/grids/100610_12ea_UEA_pm_pix.mat')['grid'], device=device)
    grid = phos_map_mat / 1080 * 512
    simu = Simulator(imgsize=512)
    img = simu(phoscoding, grid, device)

    if img is None:
        print("Check file path")
    else:
        formatted = (img * 255 / torch.max(img)).byte().cpu().numpy().squeeze()

        img = Image.fromarray(formatted)
        img.show()

if __name__ == "__main__":
    test()
