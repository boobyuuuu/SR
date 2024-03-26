import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from function.myssim import ssim_function

# 结构相似度
def batch_ssim(img1, img2):
    ssim = torch.zeros(img1.size(0))
    for i in range(img1.size(0)):
        img1_pil = to_pil_image(img1[i])
        img2_pil = to_pil_image(img2[i])
        ssim[i] = ssim_function(img1_pil, img2_pil, window_size = 16, data_range = 255.0, sigma = 1.5)
    return ssim.mean()

# 峰值信噪比
def batch_psnr(img1, img2, max_val=255.0):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)).cpu().detach().numpy()
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse)))
    return torch.mean(psnr)

class Custom_criterion(nn.Module):
    def __init__(self):
        super(Custom_criterion, self).__init__()
        self.mse_weight = 0.6
        self.ssim_weight = 0.4
        self.psnr_weight = 0
        self.l1_weight = 0

    def forward(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        ssim_loss = 1 - batch_ssim(output, target) # 取1-，因为越接近1越好
        #psnr_loss = -batch_psnr(output, target)  # 取相反数，因为 PSNR 越大越好
        #l1_loss = nn.L1Loss()(output, target)
        mse = mse_loss * self.mse_weight
        ssim = ssim_loss * self.ssim_weight
        #psnr = psnr_loss * self.psnr_weight
        #l1 = l1_loss * self.l1_weight
        return mse + ssim
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss + self.psnr_weight * psnr_loss

