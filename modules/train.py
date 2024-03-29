# 这个文件定义了train函数
import torch; torch.manual_seed(0)
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils.VAE import VAE
from utils.path_config import folder
from utils.loss import Custom_criterion1
import functions.simple_functions as simple_functions
from utils.dataset_and_dataloader import dataloader
from modules.parameter_train import EPOCHS, CUT_EPOCH, BATCH_SIZE, LATENTDIM, LR_MAX, LR_MIN

DEVICE = 'cuda'

name = f'{EPOCHS}epo_{BATCH_SIZE}bth_{LATENTDIM}latn'
vae = VAE(LATENTDIM).to(DEVICE)
vae = nn.DataParallel(vae) #将 VAE 包装成一个并行化模型，以便在多个 GPU 上并行地进行训练
criterion1 = nn.MSELoss()
criterion2 = Custom_criterion1().cuda()

def train():
    simple_functions.clearlog()
    simple_functions.log(f'{name}')
    LOSS_PLOT = []
    EPOCH_PLOT = []
    for current_epoch in range(1, EPOCHS+1):
        vae.train() # 切换成训练模式
        epoch_loss = 0.0
        # 定义优化器
        current_lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * current_epoch / EPOCHS))
        optimizer = torch.optim.AdamW(vae.parameters(), lr = current_lr)

        for _, (img_LR, img_HR) in enumerate(dataloader):
            img_LR = torch.squeeze(img_LR,dim = 1).to(DEVICE)
            img_HR = torch.squeeze(img_HR,dim = 1).to(DEVICE)
            img_SR, _, _ = vae(img_LR)
            img_SR = img_SR.to(DEVICE)
            # 这步为止，img_LR,img_HR,img_SR均是[batchsize,不知道是什么,宽，高]
            if current_epoch <= CUT_EPOCH:
                loss = criterion1(img_SR, img_HR)
            if current_epoch > CUT_EPOCH:
                loss = criterion2(img_SR, img_HR) # 每个BATCH的loss，64张图平均
            optimizer.zero_grad()
            loss.backward() # 最耗算力的一步
            optimizer.step()
            epoch_loss += loss.item()
        mean_epoch_loss = epoch_loss / len(dataloader) # 每个EPOCH的loss，全部数据集的平均
        print(f"Epoch [{current_epoch}/{EPOCHS}], Average Loss: {mean_epoch_loss:.6f}, Current_LR:{current_lr:.8f}")
        simple_functions.log(f"Epoch [{current_epoch}/{EPOCHS}], Average Loss: {mean_epoch_loss:.6f}, Current_LR:{current_lr:.8f}")

        LOSS_PLOT.append(epoch_loss)
        EPOCH_PLOT.append(current_epoch)
    fig,ax = plt.subplots()
    ax.plot(EPOCH_PLOT,LOSS_PLOT)
    ## 保存loss图片
    fig.savefig(f'{folder.run_train()}/lossfig_{name}.png', dpi = 300)
    ## 保存loss数据
    LOSS_DATA = np.stack((np.array(EPOCH_PLOT),np.array(LOSS_PLOT)),axis=0)
    np.save(f'{folder.run_train()}/lossdata_{name}.npy',LOSS_DATA)
    ## 保存模型
    torch.save(vae.state_dict(), f'{folder.run_train()}/model_{name}.pth')
    simple_functions.log(f'训练完成')