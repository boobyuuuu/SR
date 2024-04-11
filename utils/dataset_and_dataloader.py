# 这个文件定义了dataset和dataloader
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from utils.path_config import folder
from train_parameter import BATCH_SIZE, NUM_TO_LEARN_1, NUM_TO_LEARN_2

path_Confocal = folder.Confocal() 
path_Confocal_s = folder.Confocal_s() 
path_STED = folder.STED()
path_STED_s = folder.STED_s()
path_STED_HC = folder.STED_HC()

class ImageDataset_real(Dataset):
    def __init__(self, num_to_learn_1, num_to_learn_2):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.data = []
        for i in range(0, num_to_learn_1):
            img_LR_to_dataset = Image.open(f"{path_Confocal}/{i}_Confocal.png")
            img_LR_to_dataset = self.transform(img_LR_to_dataset).unsqueeze(0)
            img_HR_to_dataset = Image.open(f"{path_STED}/{i}_STED.png")
            img_HR_to_dataset = self.transform(img_HR_to_dataset).unsqueeze(0)
            self.data.append((img_LR_to_dataset, img_HR_to_dataset))
        for i in range(0, num_to_learn_2):
            img_LR_to_dataset = Image.open(f"{path_Confocal_s}/{i}.png")
            img_LR_to_dataset = self.transform(img_LR_to_dataset).unsqueeze(0)
            img_HR_to_dataset = Image.open(f"{path_STED_s}/{i}.png")
            img_HR_to_dataset = self.transform(img_HR_to_dataset).unsqueeze(0)
            self.data.append((img_LR_to_dataset, img_HR_to_dataset))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

dataset = ImageDataset_real(num_to_learn_1 = NUM_TO_LEARN_1, num_to_learn_2 = NUM_TO_LEARN_2)
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)