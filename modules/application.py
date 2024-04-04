# 这个文件定义了application函数
# 这个文件应该还能再美化
import re
import os
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch; torch.manual_seed(0)

from utils.VAE import VAE
from utils.path_config import folder
from utils.loss import mse, ssim
from utils.reconstruct import reconstrust
from application_parameter import BATCH_SIZE, EPOCHS, LATENTDIM, DO_SAVE

DEVICE = torch.device("cuda")
ITERATION = 1 # 不要改
train_list = [1500, 3000]
test_list = [5500, 6000]


def application(MODE, NAME = None):
    if MODE == 'parameter':
        batch_size, epochs, latentdim, do_save = BATCH_SIZE, EPOCHS, LATENTDIM, DO_SAVE
        model = f'{folder.output_train()}/model_{epochs}epo_{batch_size}bth_{latentdim}latn.pth'
    elif MODE == 'demo':
        target_folder = f'{folder.manual_saves()}/{NAME}'
        model_name = [file for file in os.listfolder(target_folder) if file.endswith(".pth")][0]
        model = f'{target_folder}/{model_name}'
        epochs = int(re.findall(r'\d+', model_name)[0])
        batch_size = int(re.findall(r'\d+', model_name)[1])
        latentdim = int(re.findall(r'\d+', model_name)[2])
        do_save = 0


    with torch.no_grad(): # 不要输出
        vae = VAE(latentdim).to(DEVICE)
        vae = nn.DataParallel(vae) # 并行运算带来的修饰vae的代码
        vae.load_state_dict(torch.load(model, map_location = DEVICE)) 
        vae.eval()

    for i in train_list + test_list:
        type = 'Trained' if i < 5000 else 'Test' if i > 5200 else 'Error'
        img_Confocal = Image.open(f"{folder.Confocal()}/{i}_Confocal.png")
        img_SR = reconstrust(img_Confocal, vae, ITERATION)
        img_STED = Image.open(f"{folder.STED()}/{i}_STED.png")
        img_STED_HC = Image.open(f"{folder.STED_HC()}/{i}_STED_HC.png")

        mse_loss = mse(np.array(img_SR), np.array(img_STED))
        ssim_loss = ssim(np.array(img_SR), np.array(img_STED))
        fig,ax = plt.subplots(2, 2)
        fig.suptitle(f'No.{i}, from {type}, mse = {mse_loss:.2f}, ssim = {ssim_loss:.3f}')
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
            plt.savefig(f'{folder.output_application()}/{i}.png')
            img_Confocal.save(f'{folder.output_application()}/{i}_Confocal.png')
            img_SR.save(f'{folder.output_application()}/{i}_SR.png')
            img_STED.save(f'{folder.output_application()}/{i}_STED.png')
            img_STED_HC.save(f'{folder.output_application()}/{i}_STED_HC.png')

    fig, axs = plt.subplots(4, 4, figsize=(20, 25))  # 创建一个5x4的子图网格
    r = 0
    for i in train_list + test_list:
        img_Confocal = Image.open(f"{folder.Confocal()}/{i}_Confocal.png")
        img_SR = reconstrust(img_Confocal, vae, ITERATION)
        img_STED = Image.open(f"{folder.STED()}/{i}_STED.png")
        img_STED_HC = Image.open(f"{folder.STED_HC()}/{i}_STED_HC.png")
        axs[r, 0].imshow(img_Confocal, cmap='hot')
        axs[r, 0].set_title('Confocal')
        axs[r, 1].imshow(img_SR, cmap='hot')
        axs[r, 1].set_title('Super-resolution')
        axs[r, 2].imshow(img_STED, cmap='hot')
        axs[r, 2].set_title('STED')
        axs[r, 3].imshow(img_STED_HC, cmap='hot')
        axs[r, 3].set_title('STED_HC')
        r += 1
    plt.tight_layout()  # 调整子图之间的间距
    plt.savefig(f'{folder.output_application()}/all.png')