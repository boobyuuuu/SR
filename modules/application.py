# 这个文件定义了application函数
import re
import os
import numpy as np
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch; torch.manual_seed(0)
from torchvision.transforms import transforms

from utils.path_config import folder
from utils.loss import mse, ssim, psnr, l1, l2, kl

train_list = [1500, 2000]
test_list = [5500, 6000]
train_list_s = [2000, 3000]
test_list_s = [4500, 4700]

DEVICE = 'cuda'

def reconstrust(img, model):
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

def getloss(img_SR, img_STED):
    # 不要动kl散度的顺序！
    return [mse(img_SR, img_STED), ssim(img_SR, img_STED), psnr(img_SR, img_STED), l1(img_SR, img_STED), l2(img_SR, img_STED), kl(img_STED, img_SR)]

def getimg(index, vae, mode):
    if mode == 'real':
        img_Confocal = Image.open(f"{folder.Confocal()}/{index}_Confocal.png")
        img_SR = reconstrust(img_Confocal, vae)
        img_STED = Image.open(f"{folder.STED()}/{index}_STED.png")
        img_STED_HC = Image.open(f"{folder.STED_HC()}/{index}_STED_HC.png")

        loss_list = getloss(img_SR, img_STED)
        return img_Confocal, img_SR, img_STED, img_STED_HC, loss_list
    if mode == 'simulated':
        img_Confocal = Image.open(f"{folder.Confocal_s()}/{index}.png")
        img_SR = reconstrust(img_Confocal, vae)
        img_STED = Image.open(f"{folder.STED_s()}/{index}.png")
        
        loss_list = getloss(img_SR, img_STED)
        return img_Confocal, img_SR, img_STED, loss_list

def application(mode, vae, name):
    save_folder = folder.output_application()
    if mode == 'normal':
        model = f'{folder.output_train()}/{name}'
        latentdim = int(re.findall(r'\d+', name)[-1])
    elif mode == 'demo':
        target_folder = f'{folder.manual_saves()}/{name}'
        model_name = [file for file in os.listdir(target_folder) if file.endswith(".pth")][0]
        model = f'{target_folder}/{model_name}'
        latentdim = int(re.findall(r'\d+', model_name)[-1])

    with torch.no_grad(): # 不要输出
        vae = vae(latentdim).to(DEVICE)
        vae = nn.DataParallel(vae) # 并行运算带来的修饰vae的代码
        vae.load_state_dict(torch.load(model, map_location = DEVICE)) 
        vae.eval()

    # 真实数据集画单图
    for i in train_list + test_list:
        type = 'Trained' if i < 5000 else 'Test' if i > 5200 else 'Error'
        img_Confocal, img_SR, img_STED, img_STED_HC, loss_list = getimg(i, vae, 'real')
        fig,ax = plt.subplots(2, 2)
        fig.suptitle(f'No.{i}, from {type}, mse = {loss_list[0]:.3e}, ssim = {loss_list[1]:.4f}\n psnr = {loss_list[2]:.3f}, l1 = {loss_list[3]:.3f}, l2 = {loss_list[4]:.3f}, kl = {loss_list[5]:.3f}')
        ax[0, 0].imshow(img_Confocal, cmap='hot')
        ax[0, 0].set_title('Confocal')
        ax[0, 0].axis('off')
        ax[0, 1].imshow(img_SR, cmap='hot')
        ax[0, 1].set_title('Super-resolution')
        ax[0, 1].axis('off')
        ax[1, 0].imshow(img_STED_HC, cmap='hot')
        ax[1, 0].set_title('STED_HC')
        ax[1, 0].axis('off')
        ax[1, 1].imshow(img_STED, cmap='hot')
        ax[1, 1].set_title('STED')
        ax[1, 1].axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_folder}/real_{i}.png')
        img_Confocal.save(f'{save_folder}/real_{i}_Confocal.png')
        img_SR.save(f'{save_folder}/real_{i}_SR.png')
        img_STED.save(f'{save_folder}/real_{i}_STED.png')
        img_STED_HC.save(f'{save_folder}/real_{i}_STED_HC.png')
    
    # 模拟数据集画单图
    for i in train_list_s + test_list_s:
        type = 'Trained' if i < 4000 else 'Test' 
        img_Confocal, img_SR, img_STED, loss_list = getimg(i, vae, 'simulated')
        fig,ax = plt.subplots(1,3)
        fig.suptitle(f'No.{i}, from {type}, mse = {loss_list[0]:.3e}, ssim = {loss_list[1]:.4f}\n psnr = {loss_list[2]:.3f}, l1 = {loss_list[3]:.3f}, l2 = {loss_list[4]:.3f}, kl = {loss_list[5]:.3f}')
        ax[0].imshow(img_Confocal, cmap='hot')
        ax[0].set_title('Confocal')
        ax[0].axis('off')
        ax[1].imshow(img_SR, cmap='hot')
        ax[1].set_title('Super-resolution')
        ax[1].axis('off')
        ax[2].imshow(img_STED, cmap='hot')
        ax[2].set_title('STED')
        ax[2].axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_folder}/s_{i}.png')
        img_Confocal.save(f'{save_folder}/s_{i}_Confocal.png')
        img_SR.save(f'{save_folder}/s_{i}_SR.png')
        img_STED.save(f'{save_folder}/s_{i}_STED.png')

    # 真实画组图
    fig, axs = plt.subplots(len(train_list+test_list), 4, figsize=(20, 15))  # 创建一个ix4的子图网格
    r = 0
    for i in train_list + test_list:
        type = 'Trained' if i < 5000 else 'Test' if i > 5200 else 'Error'
        img_Confocal, img_SR, img_STED, img_STED_HC, loss_list = getimg(i, vae, 'real')
        axs[r, 0].imshow(img_Confocal, cmap='hot')
        axs[r, 0].set_title(f'NO.{i},from {type}\n\nConfocal')
        axs[r, 1].imshow(img_SR, cmap='hot')
        axs[r, 1].set_title(f'mse = {loss_list[0]:.3e}, ssim = {loss_list[1]:.4f}\n\nSuper-resolution')
        axs[r, 2].imshow(img_STED, cmap='hot')
        axs[r, 2].set_title(f'psnr = {loss_list[2]:.3f}, l1 = {loss_list[3]:.3f}\n\nSTED')
        axs[r, 3].imshow(img_STED_HC, cmap='hot')
        axs[r, 3].set_title(f'l2 = {loss_list[4]:.3f}, kl = {loss_list[5]:.3f}\n\nSTED_HC')
        r += 1
    plt.tight_layout()  # 调整子图之间的间距
    plt.savefig(f'{save_folder}/real_all.png')

    # 模拟画组图
    fig, axs = plt.subplots(len(train_list_s+test_list_s), 3, figsize=(20, 15))  # 创建一个ix3的子图网格
    r = 0
    for i in train_list_s + test_list_s:
        type = 'Trained' if i < 4000 else 'Test' 
        img_Confocal, img_SR, img_STED, loss_list = getimg(i, vae, 'simulated')
        axs[r, 0].imshow(img_Confocal, cmap='hot')
        axs[r, 0].set_title(f'NO.{i},from {type}\n\nConfocal')
        axs[r, 1].imshow(img_SR, cmap='hot')
        axs[r, 1].set_title(f'mse = {loss_list[0]:.3e}, ssim = {loss_list[1]:.4f}, psnr = {loss_list[2]:.3f}\n\nSuper-resolution')
        axs[r, 2].imshow(img_STED, cmap='hot')
        axs[r, 2].set_title(f'l1 = {loss_list[3]:.3f}, l2 = {loss_list[4]:.3f}, kl = {loss_list[5]:.3f}\n\nSTED')
        r += 1
    plt.tight_layout()  # 调整子图之间的间距
    plt.savefig(f'{save_folder}/simulated_all.png')