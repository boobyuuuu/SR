# 这个文件定义了自定义的评估标准
import torch; torch.manual_seed(0)
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

from functions.custom_ssim import custom_ssim

# 下面出现的img1和img2都是第0维为batch内index的四维张量
# 计算一个batch的平均结构相似度ssim
def batch_ssim(img1, img2):
    ssim = torch.zeros(img1.size(0))
    for i in range(img1.size(0)):
        img1_pil = to_pil_image(img1[i])
        img2_pil = to_pil_image(img2[i])
        ssim[i] = custom_ssim(img1_pil, img2_pil, window_size = 16, data_range = 255.0, sigma = 1.5)
    return ssim.mean()

# 计算一个batch的平均峰值信噪比psnr
def batch_psnr(img1, img2, max_val=255.0):
    mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)).cpu().detach().numpy()
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(torch.tensor(mse)))
    return torch.mean(psnr)

class Custom_criterion1(nn.Module):
    def __init__(self):
        super(Custom_criterion1, self).__init__()
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

