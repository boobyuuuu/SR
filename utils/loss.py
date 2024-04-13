# 这个文件定义了自定义的评估标准
import torch; torch.manual_seed(0)
import torch.nn as nn
import numpy as np
from torchvision.transforms.functional import to_pil_image

from functions.custom_ssim import custom_ssim
from functions.custom_kl import custom_kl

# pic2pic计算
def mse(img1, img2):
    return np.mean((img1 - img2)**2)/(255**2)

def ssim(img1, img2):
    return custom_ssim(img1, img2, window_size = 16, data_range = 255.0, sigma = 1.5)

def psnr(img1, img2):
    return 10 * np.log10((256 ** 2) / mse(img1, img2))

def l1(img1, img2):
    img1_n = img1 / 255.0
    img2_n = img2 / 255.0
    return np.sum(np.square(img1_n - img2_n))

def l2(img1, img2):
    img1_n = img1 / 255.0
    img2_n = img2 / 255.0
    return np.sum(np.abs(img1_n - img2_n))

def kl(img1, img2):
    return custom_kl(img1, img2)/ (255**2)

# batch2batch计算
def batch_ssim(img1, img2):
    ssim_loss = torch.zeros(img1.size(0))
    for i in range(img1.size(0)):
        img1_pil = to_pil_image(img1[i])
        img2_pil = to_pil_image(img2[i])
        ssim_loss[i] = ssim(img1_pil, img2_pil)
    return ssim_loss.mean()

def batch_kl(img1, img2):
    kl_loss = torch.zeros(img1.size(0))
    for i in range(img1.size(0)):
        img1_pil = to_pil_image(img1[i])
        img2_pil = to_pil_image(img2[i])
        kl_loss[i] = kl(img1_pil, img2_pil)
    return kl_loss.mean()

class Custom_criterion1(nn.Module):
    def __init__(self):
        super(Custom_criterion1, self).__init__()
        self.mse_weight = 0.7
        self.kl_weight = 0.3
        self.ssim_weight = 0.3

    def forward(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        #kl_loss = batch_kl(target,output)
        ssim_loss = 1 - batch_ssim(output, target) # 取1-，因为越接近1越好
        #psnr_loss = -batch_psnr(output, target)  # 取相反数，因为 PSNR 越大越好
        #l1_loss = nn.L1Loss()(output, target)

        mse = mse_loss * self.mse_weight
        ssim = ssim_loss * self.ssim_weight
        #kl = kl_loss * self.kl_weight
        #psnr = psnr_loss * self.psnr_weight
        #l1 = l1_loss * self.l1_weight
        return mse + ssim

