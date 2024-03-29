# 主要import
import torch; torch.manual_seed(0)
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
# 模块import
from utils.VAE import VAE
from utils.path_config import folder
from parameter_application import BATCH_SIZE, EPOCHS, LATENTDIM, DO_SAVE

DEVICE = torch.device("cuda")
ITERATION = 1 # 不要改
train_list = [1500, 3000]
test_list = [5500, 6000]



def reconstrust(img):
    for i in range(ITERATION):
        preprocess = transforms.Compose([transforms.ToTensor()])
        img= preprocess(img)
        img = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            img = vae(img)
        img = img[0][0].squeeze(0)
        img = img.to('cpu').numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
    return img

def application(MODE):
    if MODE == application:
        model = f'{folder.run_train()}/model_{EPOCHS}epo_{BATCH_SIZE}bth_{LATENTDIM}latn.pth'
    else:
        MODE =

    with torch.no_grad(): # 不要输出
    vae = VAE(LATENTDIM).to(DEVICE)
    vae = nn.DataParallel(vae) # 并行运算带来的修饰vae的代码
    vae.load_state_dict(torch.load(model, map_location=DEVICE)) 
    vae.eval()

    for i in train_list + test_list:
        type = '训练集' if i < 5000 else '测试集' if i > 5200 else '出问题了'
        img_Confocal, img_STED, img_STED_HC = Image.open(f"{folder.Confocal()}/{i}_Confocal.png"), Image.open(f"{folder.STED()}/{i}_STED.png") ,Image.open(f"{folder.STED_HC()}/{i}_STED_HC.png")
        img_SR = reconstrust(img_Confocal)

        fig,ax = plt.subplots(1,4)
        fig.suptitle(f'No.{i},来自 type')
        ax[0].imshow(img_Confocal,cmap='hot')
        ax[0].set_title('Confocal')
        ax[1].imshow(img_SR,cmap='hot')
        ax[1].set_title('Super-resolution')
        ax[2].imshow(img_STED,cmap='hot')
        ax[2].set_title('STED')
        ax[3].imshow(img_STED_HC,cmap='hot')
        ax[3].set_title('STED_HC')
        if DO_SAVE == 1:
            plt.savefig(f'{folder.run_application()}/{i}.png')
            img_Confocal.save(f'{folder.run_application()}/{i}_Confocal.png')
            img_SR.save(f'{folder.run_application()}/{i}_SR.png')
            img_STED.save(f'{folder.run_application()}/{i}_STED.png')
            img_STED_HC.save(f'{folder.run_application()}/{i}_STED_HC.png')