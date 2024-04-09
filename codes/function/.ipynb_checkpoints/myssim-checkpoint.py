import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

def ssim_function(img1, img2, window_size=11, data_range=255.0, sigma=1.5):
    K1 = 0.01
    K2 = 0.03

    # 将图像转换为numpy数组
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)

    # 计算SSIM的常数
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 高斯滤波
    window = np.outer(gaussian(window_size, sigma), gaussian(window_size, sigma))

    # 计算均值
    mu1 = convolve2d(img1, window, mode='valid')
    mu2 = convolve2d(img2, window, mode='valid')

    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 ** 2, window, mode='valid') - mu1_sq
    sigma2_sq = convolve2d(img2 ** 2, window, mode='valid') - mu2_sq
    sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    # 计算SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    # 返回平均SSIM
    return np.mean(ssim_map)

def gaussian(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()
