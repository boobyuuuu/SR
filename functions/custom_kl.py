import numpy as np

def custom_kl(img1, img2, eps=1e-10):
    p = image_to_distribution(img1) + eps
    q = image_to_distribution(img2) + eps
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def image_to_distribution(image):
    image_n = image / 255.0
    # 计算像素值分布（概率分布）
    distribution = np.histogram(image_n.flatten(), bins=256, range=(0, 1), density=True)[0]
    return distribution