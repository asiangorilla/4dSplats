import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# L1 loss
def l1_loss(img1, img2):
    return torch.abs((img1 - img2)).mean()


# SSIM Loss
def compute_ssim(img1, img2, window_size=11, size_average=True):

    channel = img1.size(-3)  # Get the number of channels, RGB or RGBA
    window = create_window(window_size, channel)  # Create a Gaussian window filter

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)  # Calculate SSIM using the defined window


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # 1D Gaussian and add a dimension
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  # Create a 2D Gaussian Matrix(window) by outer product WW^T

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())  # Match the number of image channels
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])  # Gaussian Function
    return gauss / gauss.sum()  # Normalize

def _ssim(img1, img2, window, window_size, channel):

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)  # img1 convolutoin, window --> convolution kernel
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)  # img2 convolution

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq  # sigma^2 = E[(x - mu)^2] = E[x^2] - E[x]^2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2  # Covariance

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))  # SSIM Formulas

    return ssim.mean()

