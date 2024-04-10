# 这个文件定义了application函数
# 这个文件应该还能再美化
import re
import os
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch; torch.manual_seed(0)

from utils.path_config import folder
from utils.loss import mse, ssim, psnr, l1, l2, kl
from utils.reconstruct import reconstrust
from application_parameter import BATCH_SIZE, EPOCHS, LATENTDIM, DO_SAVE

DEVICE = torch.device("cuda")
ITERATION = 1 # 不要改
train_list = [1500, 3000]
test_list = [5500, 6000]


def application(MODE, vae, NAME = None):
    if MODE == 'parameter':
        batch_size, epochs, latentdim, do_save = BATCH_SIZE, EPOCHS, LATENTDIM, DO_SAVE
        model = f'{folder.output_train()}/model_{epochs}epo_{batch_size}bth_{latentdim}latn.pth'
        save_folder = folder.output_application()
    elif MODE == 'demo':
        target_folder = f'{folder.manual_saves()}/{NAME}'
        model_name = [file for file in os.listdir(target_folder) if file.endswith(".pth")][0]
        model = f'{target_folder}/{model_name}'
        epochs = int(re.findall(r'\d+', model_name)[-3])
        batch_size = int(re.findall(r'\d+', model_name)[-2])
        latentdim = int(re.findall(r'\d+', model_name)[-1])
        do_save = 1
        save_folder = folder.output_demo()


    with torch.no_grad(): # 不要输出
        vae = vae(latentdim).to(DEVICE)
        vae = nn.DataParallel(vae) # 并行运算带来的修饰vae的代码
        vae.load_state_dict(torch.load(model, map_location = DEVICE)) 
        vae.eval()

    for i in train_list + test_list:
        type = 'Trained' if i < 5000 else 'Test' if i > 5200 else 'Error'
        img_Confocal = Image.open(f"{folder.Confocal()}/{i}_Confocal.png")
        img_SR = reconstrust(img_Confocal, vae, ITERATION)
        img_STED = Image.open(f"{folder.STED()}/{i}_STED.png")
        img_STED_HC = Image.open(f"{folder.STED_HC()}/{i}_STED_HC.png")

        img_SR_np = np.array(img_SR)
        img_STED_np = np.array(img_STED)

        mse_loss = mse(img_SR_np, img_STED_np)
        ssim_loss = ssim(img_SR_np, img_STED_np)
        psnr_loss = psnr(img_SR_np, img_STED_np)
        l1_loss = l1(img_SR_np, img_STED_np)
        l2_loss = l2(img_SR_np, img_STED_np)
        kl_loss = kl(img_STED_np, img_SR_np) # 尤其注意这个的顺序
        fig,ax = plt.subplots(2, 2)
        fig.suptitle(f'No.{i}, from {type}, mse = {mse_loss:.3e}, ssim = {ssim_loss:.4f}\n psnr = {psnr_loss:.3f}, l1 = {l1_loss:.3f}, l2 = {l2_loss:.3f}, kl = {kl_loss:.3f}')
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
        if do_save == 1:
            plt.savefig(f'{save_folder}/{i}.png')
            img_Confocal.save(f'{save_folder}/{i}_Confocal.png')
            img_SR.save(f'{save_folder}/{i}_SR.png')
            img_STED.save(f'{save_folder}/{i}_STED.png')
            img_STED_HC.save(f'{save_folder}/{i}_STED_HC.png')

    fig, axs = plt.subplots(4, 4, figsize=(20, 25))  # 创建一个4x4的子图网格
    r = 0
    for i in train_list + test_list:
        type = 'Trained' if i < 5000 else 'Test' if i > 5200 else 'Error'
        img_Confocal = Image.open(f"{folder.Confocal()}/{i}_Confocal.png")
        img_SR = reconstrust(img_Confocal, vae, ITERATION)
        img_STED = Image.open(f"{folder.STED()}/{i}_STED.png")
        img_STED_HC = Image.open(f"{folder.STED_HC()}/{i}_STED_HC.png")
    
        img_SR_np = np.array(img_SR)
        img_STED_np = np.array(img_STED)

        mse_loss = mse(img_SR_np, img_STED_np)
        ssim_loss = ssim(img_SR_np, img_STED_np)
        psnr_loss = psnr(img_SR_np, img_STED_np)
        l1_loss = l1(img_SR_np, img_STED_np)
        l2_loss = l2(img_SR_np, img_STED_np)
        kl_loss = kl(img_STED_np, img_SR_np) 

        axs[r, 0].imshow(img_Confocal, cmap='hot')
        axs[r, 0].set_title(f'NO.{i},from {type}\n\nConfocal')
        axs[r, 1].imshow(img_SR, cmap='hot')
        axs[r, 1].set_title(f'mse = {mse_loss:.3e}, ssim = {ssim_loss:.4f}\n\nSuper-resolution')
        axs[r, 2].imshow(img_STED, cmap='hot')
        axs[r, 2].set_title(f'psnr = {psnr_loss:.3f}, l1 = {l1_loss:.3f}\n\nSTED')
        axs[r, 3].imshow(img_STED_HC, cmap='hot')
        axs[r, 3].set_title(f'l2 = {l2_loss:.3f}, kl = {kl_loss:.3f}\n\nSTED_HC')
        r += 1
    plt.tight_layout()  # 调整子图之间的间距
    plt.savefig(f'{save_folder}/all.png')