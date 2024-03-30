# 这个文件定义了单张图片重建函数
import numpy as np
from PIL import Image
import torch; torch.manual_seed(0)
from torchvision.transforms import transforms

DEVICE = 'cuda'

def reconstrust(img, model, iteration):
    for i in range(iteration):
        preprocess = transforms.Compose([transforms.ToTensor()])
        img= preprocess(img)
        img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img = model(img)
        img = img[0][0].squeeze(0)
        img = img.to('cpu').numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
    return img