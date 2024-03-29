# 这个文件将datasets准备好了
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


from utils.path_config import folder
from modules.train import BATCH_SIZE
from modules.train import NUM_TO_LEARN
from modules.train import MODE

path_Confocal = folder.Confocal() 
path_STED = folder.STED()
path_STED_HC = folder.STED_HC()

class ImageDataset(Dataset):
    def __init__(self, num_to_learn, mode):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data = []
        if mode == 0:
            path_HR = path_STED_HC
            name = '_STED_HC'
        elif mode == 1:
            path_HR = path_STED
            name = '_STED'
        for i in range(0, num_to_learn):
            img_LR_to_dataset = Image.open(f"{path_Confocal}/{i}_Confocal.png")
            img_LR_to_dataset = self.transform(img_LR_to_dataset).unsqueeze(0)
            img_HR_to_dataset = Image.open(f"{path_HR}/{i}{name}.png")
            img_HR_to_dataset = self.transform(img_HR_to_dataset).unsqueeze(0)
            self.data.append((img_LR_to_dataset, img_HR_to_dataset))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    
dataset = ImageDataset(NUM_TO_LEARN, MODE)
dataloader = DataLoader(dataset, BATCH_SIZE, True)