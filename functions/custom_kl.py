# 这个文件定义了一个计算两张图片kl散度的函数
import numpy as np

# 输入的img为Image库打开的图像
def custom_kl(img1, img2, eps=1e-10):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)

    distribution1 = np.histogram(img1.flatten(), bins=256, range=(0, 1), density=True)[0]
    distribution2 = np.histogram(img2.flatten(), bins=256, range=(0, 1), density=True)[0]

    p = distribution1 + eps
    q = distribution2 + eps
    
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))